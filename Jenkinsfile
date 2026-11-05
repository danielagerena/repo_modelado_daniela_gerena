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
                // Placeholder: en 1b.3 se conecta aqui el merge real develop -> master
                // usando un token de GitHub guardado como credencial de Jenkins.
                echo 'APROBADO. (Pendiente 1b.3: merge real develop -> master con token de GitHub.)'
            }
        }
    }

    post {
        success {
            echo "Build EXITOSO. Resultado: ${currentBuild.currentResult}"
        }
        failure {
            echo "Build FALLIDO. Resultado: ${currentBuild.currentResult}"
        }
        always {
            echo "Fin del pipeline. Estado final: ${currentBuild.currentResult}"
        }
    }
}