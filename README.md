# NYU Langone Medical Center Knee Alignment Research

@author Jack Bosco

---

### What is the Goal?

Using machine learning techniques such as feature selection and data clustering, we hope to develop our understanding of knee alignment morphologies.

### Where is the data?

Due to compliance reasons, I cannot upload the datasheet to GitHub.
However, if you have the `mako_data.xlsx` file, drop that in `raw`.

### How do I run the project?

1. `cd` to the project directory
2. Make sure you have the right dependencies by running the command below. You only need to do this once.
   ```
   pip3 install -r requirements.txt
   ```
3. To treat the data, run the command below. This creates the treated spreadsheet `treated/morphologies.csv`. You only need to run this once.
   ```
   python3 treat_data.py
   ```
4. Visualize the treated data by running
   ```
   python3 data_viz.py
   ```
5. Create a data clustering example (optional command line options are `nclusters <int>`, `bmi`, `age`, `FTR` for femoral transverse rotation, `sex`):
   ```
   python3 make_clusters.py
   ```
5. Create and visualize a regression model for planned aHKA
   ```
   python3 regression.py
   ```

### Config options

Configure the date file locations in `config.py`:
 - `raw_path` is the path to the raw data
 - `treated_path` is the path to the treated data

Of course, if the files are not there the program will just crash.
I also cannot privide the files in this repo due to compliance reasons, though please reach out to me if you would like to run this on your own dataset.