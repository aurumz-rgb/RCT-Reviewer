As per plan:

rct_reviewer/


app.py : will use joblib files

app1.py : will use pickle files

app2.py : will connect with hugging face hub.






To run scripts/pickle-to-joblib:

python convert_models.py 

in the .venv of the same directory






To upload the files to hugging face, use: (it uploaded h5) files too so make sure you rmeove that by either modifying the upload_to_hf.py or remove the h5 after committing.

python upload_to_hf.py






What to do now?


simple, clean the files acche se and push and check 