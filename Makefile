test_api:
	cd src && uvicorn api:app --reload

streamlit:
	streamlit run streamlit_app.py

docker:
	docker-compose up
