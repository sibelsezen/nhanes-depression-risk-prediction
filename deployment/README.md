# AWS EC2 Deployment

This guide reproduces the deployment of the NHANES Streamlit application on an Ubuntu EC2 instance.

## 1. EC2 and network configuration

Create an Ubuntu EC2 instance with:

- Python 3 support
- SSH access restricted to your IP address
- Inbound TCP port `8501` restricted to the intended audience
- Enough memory to install the scientific Python dependencies

For a public production deployment, prefer HTTPS through a reverse proxy rather than exposing Streamlit directly.

## 2. Connect and install system packages

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip
```

## 3. Clone and install the application

```bash
cd /home/ubuntu
git clone https://github.com/sibelsezen/nhanes-depression-risk-prediction.git
cd nhanes-depression-risk-prediction

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Confirm that the saved model exists:

```bash
ls -lh models/final_depression_risk_model.joblib
```

Test the application manually:

```bash
python3 -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
```

## 4. Install the systemd service

```bash
sudo cp deployment/nhanes-streamlit.service \
  /etc/systemd/system/nhanes-streamlit.service

sudo systemctl daemon-reload
sudo systemctl enable --now nhanes-streamlit.service
```

Check the service:

```bash
sudo systemctl status nhanes-streamlit.service
sudo journalctl -u nhanes-streamlit.service -n 100 --no-pager
```

## 5. Update a running deployment

```bash
cd /home/ubuntu/nhanes-depression-risk-prediction
git pull
source .venv/bin/activate
python3 -m pip install -r requirements.txt
sudo systemctl restart nhanes-streamlit.service
sudo systemctl status nhanes-streamlit.service
```

## Security notes

- Never commit an EC2 private key, `.pem` file, password, or Streamlit secrets file.
- Restrict SSH and application ports in the EC2 security group.
- Use an Elastic IP or domain if the endpoint must remain stable.
- For a production-facing service, terminate HTTPS with Nginx or an AWS load balancer.
- The application is an educational screening demonstration, not a clinical service.
