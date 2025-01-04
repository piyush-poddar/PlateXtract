# PlateXtract

Extract text from vehicle number plates with ease and over 99% accuracy.

The application is already deployed on Streamlit. Check it out here **https://platextract.streamlit.app**

If you want it on your system do the following:

- Clone this github repo.
- Run `pip install -r requirements.txt` in any terminal.
- Download model weights from this [link](https://drive.google.com/uc?id=1Qlcv7vcyWn9UsKsjqHat4V_CuVh5Lggs) and put it inside `model/weights` directory. Alternatively, you can skip this step allow the code itself to do it for you.
- Add your Gemini API key in `.streamlit/secrets.toml`.
- Run `streamlit run app.py` and just see the magic.
