Requirements

    python 3.12

    numpy==2.1.1
    pandas==2.2.2
    sklearn==1.5.2

Running:

    To run the demo, execute:
        python predict.py 

    After running the script in that folder will be generated <prediction_results.csv> 
    The file has 'Sleep_Quality' column with the result value.

    The input is expected  csv file in the same folder with a name <new_data.csv>. The file should have all features columns. 

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain all features columns and target for prediction Churn.
    After running the script the "param_dict.pickle"  and "finalized_model.saw" will be created.
    Run the training script:
        python train.py
