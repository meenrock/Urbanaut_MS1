pipeline {
    agent any

    environment {
        VENV_DIR = 'quickstart'
    }

    stages {
        stage('System Setup') {
            steps {
                sh '''
                sudo apt update
                sudo apt install python3 python3-pip python3-venv -y
                '''
            }
        }

        stage('Set Up Virtual Environment') {
            steps {
                sh '''
                rm -rf $VENV_DIR
                python3 -m venv $VENV_DIR
                '''
            }
        }

        stage('Activate Environment & Install Requirements') {
            steps {
                sh '''
                source $VENV_DIR/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Serve BentoML Service') {
            steps {
                sh '''
                source $VENV_DIR/bin/activate
                nohup bentoml serve > bentoml.log 2>&1 &
                '''
            }
        }
    }

    post {
        success {
            echo '✅ BentoML service setup and serving started successfully.'
        }
        failure {
            echo '❌ Failed to start BentoML service.'
        }
    }
}
