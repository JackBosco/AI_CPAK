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
