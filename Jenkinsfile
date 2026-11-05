// Jenkinsfile — Pipeline de aprobacion para repo_modelado_daniela_gerena
//
// Entregable Final CDP, Fase 1. El pipeline:
//   1. Clona el repositorio (checkout de la rama configurada en el job).
//   2. Prepara un entorno virtual e instala las dependencias.
//   3. Corre la prueba de estructura (carpetas/archivos clave).
//   4. Corre la prueba de metricas (gate por PR-AUC) y archiva metricas.json.
//   5. Se detiene en un gate manual de aprobacion antes de integrar.
//   6. (1b.3) Integrara develop en master tras la aprobacion.
//
// El job se configura como "Pipeline script from SCM" apuntando a este archivo,
// de modo que el Jenkinsfile quede versionado en el propio repositorio.

// Envia una notificacion a un webhook (Discord) al terminar el pipeline.
// La URL se guarda como credencial 'webhook-url' (Secret text) en Jenkins.
// El envio es best-effort: si el webhook falla, no se tumba el build.
def notificar(String mensaje) {
    withCredentials([string(credentialsId: 'webhook-url', variable: 'WEBHOOK')]) {
        withEnv(["MENSAJE=${mensaje}"]) {
            sh '''
                python3 - <<'PY'
import json, os, urllib.request
url = os.environ["WEBHOOK"]
payload = json.dumps({"content": os.environ["MENSAJE"]}).encode("utf-8")
req = urllib.request.Request(
    url, data=payload, headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=15)
    print("Notificacion enviada.")
except Exception as e:
    print("No se pudo enviar la notificacion:", e)
PY
            '''
        }
    }
}

pipeline {
    agent any

    options {
        // Evita que un gate de aprobacion olvidado deje el build colgado para siempre.
        timeout(time: 2, unit: 'HOURS')
        disableConcurrentBuilds()
    }

    environment {
        VENV = 'venv_ci'
    }

    stages {

        stage('Clonar repo') {
            steps {
                echo 'Paso 1: clonando el repositorio (checkout de la rama configurada)...'
                checkout scm
            }
        }

        stage('Preparar entorno') {
            steps {
                echo 'Paso 2: creando entorno virtual e instalando dependencias...'
                sh '''
                    python3 -m venv "${VENV}"
                    . "${VENV}/bin/activate"
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Prueba de estructura') {
            steps {
                echo 'Paso 3: validando la estructura de carpetas y archivos...'
                sh '''
                    . "${VENV}/bin/activate"
                    python tests/test_structure.py
                '''
            }
        }

        stage('Prueba de metricas') {
            steps {
                echo 'Paso 4: evaluando el modelo y aplicando el gate de PR-AUC...'
                sh '''
                    . "${VENV}/bin/activate"
                    python mlops_pipeline/src/evaluate.py
                '''
            }
            post {
                // Guarda metricas.json como artefacto del build (evidencia para el informe),
                // haya pasado o no el gate, para poder revisar el valor obtenido.
                always {
                    archiveArtifacts artifacts: 'mlops_pipeline/reportes/metricas.json',
                                     allowEmptyArchive: true
                }
            }
        }

        stage('Aprobacion') {
            steps {
                echo 'Paso 5: esperando aprobacion manual para integrar...'
                timeout(time: 1, unit: 'HOURS') {
                    input message: 'Las pruebas pasaron. ¿Integrar develop en master?',
                          ok: 'Aprobar integracion'
                }
            }
        }

        stage('Integrar a master') {
            steps {
                echo 'Paso 6: integrando develop en master tras la aprobacion...'
                // La credencial 'github-token' (usuario + token) se define en Jenkins.
                // Jenkins enmascara GIT_TOKEN en el log automaticamente.
                withCredentials([usernamePassword(
                        credentialsId: 'github-token',
                        usernameVariable: 'GIT_USER',
                        passwordVariable: 'GIT_TOKEN')]) {
                    sh '''
                        set -e
                        git config user.email "danigerena.lopera@gmail.com"
                        git config user.name  "Jenkins CI"

                        # Trae ambas ramas de forma explicita (el checkout del job solo
                        # garantiza la rama construida).
                        git fetch origin \
                            +refs/heads/master:refs/remotes/origin/master \
                            +refs/heads/develop:refs/remotes/origin/develop

                        # Posiciona master local en el estado remoto y fusiona develop.
                        git checkout -B master origin/master
                        git merge --no-ff origin/develop \
                            -m "CI: integra develop en master tras aprobacion (build ${BUILD_NUMBER})"

                        # Publica master usando el token (queda enmascarado en el log).
                        git push "https://${GIT_USER}:${GIT_TOKEN}@github.com/danielagerena/repo_modelado_daniela_gerena.git" master
                    '''
                }
                echo 'Integracion completada: master actualizado con los cambios de develop.'
            }
        }
    }

    post {
        success {
            notificar("✅ ${JOB_NAME} #${BUILD_NUMBER}: build EXITOSO. develop integrado a master. ${BUILD_URL}")
        }
        failure {
            notificar("❌ ${JOB_NAME} #${BUILD_NUMBER}: build FALLIDO. Revisar: ${BUILD_URL}")
        }
        aborted {
            notificar("⚠️ ${JOB_NAME} #${BUILD_NUMBER}: build ABORTADO. Integracion no aprobada. ${BUILD_URL}")
        }
        always {
            echo "Fin del pipeline. Estado final: ${currentBuild.currentResult}"
        }
    }
}