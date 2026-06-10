
<please ignore this file.>


# V.S.


rct_reviewer/

app.py : uses .joblib, .npz files (locally have to pull then model files using LFS)

app1.py : uses .pickle, .pck, .npz files (locally have to pull then model files using LFS)

app2.py : works by connecting with hugging face hub repository [Aurumz/RCT-Reviewer] (can be run directly without pulling the model files uing LFS)




- To run scripts/pickle-to-joblib in the .venv of the same directory:

python convert_models.py 




- This uploads the model files to Hugging Face Hub.

python upload_to_hf.py



- P.S: Also check the git LFS pull / push, commit guide .md file.