test_api:
	cd src && uvicorn api:app --reload

streamlit:
	streamlit run streamlit_app.py

docker:
	docker-compose up


# Note to self: Yes these are secrets, don't commit the PEM file to the repo
f_pem = 'aws_streamlit.pem'

# AWS IP address needs to be manually adjusted
IP = 54.91.67.112
username = ubuntu

ssh:
	chmod 600 $(f_pem)
	ssh -i $(f_pem) -o "StrictHostKeyChecking no" $(username)@$(IP)
