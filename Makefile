test_api:
	cd src && uvicorn api:app --reload

streamlit_topic:
	streamlit run streamlit_app.py
#
#streamlit_explain:
#	streamlit run streamlit_app_explainer.py
