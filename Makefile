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
dockertag = nchsmalletdockerized_nchs-mallet-api

ssh:
	chmod 600 $(f_pem)
	ssh -i $(f_pem) -o "StrictHostKeyChecking no" $(username)@$(IP)

update:
	scp -r -i $(f_pem) ../NCHS-Mallet-Dockerized $(username)@$(IP):~


update_image:
	docker save $(dockertag) | bzip2 | pv | ssh -i $(f_pem) $(username)@$(IP) 'bunzip2 | docker load'
