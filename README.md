# ML Job End Time Prediction using SLURM Logs

## Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sanket5153/Monitoring_Portal.git
cd Monitoring_Portal
```

### Step 2: Install Miniconda (if not already installed)
Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

### Step 3: Create and Activate Conda Environment
```bash
conda create -n ml_env python=3.9 -y
conda activate ml_env
```

### Step 4: Install Required Packages
Ensure you're in the project directory and run:
```bash
pip install -r requirements.txt
```

---

## Project Workflow

1. **Data Ingestion**
   - The script `data_ingestion.py` reads data from a CSV file.
   - It splits the data into `train.csv` and `test.csv` and stores them in the `artifacts/` directory.
   - Then it calls `data_transformation.py`.

2. **Data Transformation**
   - The `data_transformation.py` script performs preprocessing on the data.
   - It saves the preprocessor object as `preprocessor.pkl` in the `artifacts/` directory.

3. **Model Training**
   - The `model_trainer.py` script is triggered next.
   - It trains the machine learning model and stores the trained model as `model.pkl` inside the `artifacts/` directory.

To run the full pipeline:
```bash
python data_ingestion.py
```

---

## Real-Time Job End Time Prediction

1. Open the `real_time_prediction.py` script.
2. Modify it according to your SLURM database setup:
   - Configure database connection.
   - Provide the **Job ID** whose end time you want to predict.

3. Run the prediction script:
```bash
python real_time_prediction.py
```

4. The script will output the predicted **job end time**.
