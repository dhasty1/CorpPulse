## CorpPulse

You can view this project live on [Streamlit Community Cloud](https://corppulse.streamlit.app/)!

### API Key
You will need to sign up for an OpenAI API key at (OpenAI)[https://openai.com/index/openai-api]. Use this in place of `OPENAI_KEY = os.getenv("OPENAI_KEY")` within pipeline.py to access the LLM.

### Installation
To install this application on your local machine, clone the repository and install the necessary dependencies.

```bash
pip install -r requirements.txt
```

After the dependencies have been installed, you can run the application.

```bash
streamlit run app.py
```

### Features

This application serves as a chatbot leveraging OpenAI's GPT 3.5-Turbo LLM to query against a set of shareholder letters. Users can also upload their own .txt documents to query against!

### Try it out

Ask the chatbot "Identify macroeconomic trends in financial companies" and watch as the magic happens!
