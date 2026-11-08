# My 4th Year Project: E-commerce Customer Behavior Model

Hey there! This is my final-year computer science capstone project. I built a full-stack machine learning pipeline that looks at how people browse an online store and predicts whether they are actually going to buy something or just window-shopping. 

The goal was to replace slow, end-of-the-month sales spreadsheets with something that watches user behavior in real-time and acts on it instantly.

*   **My Model's Accuracy:** 77.15% (Tested on 10,000 hidden rows)
*   **The Stack I Used:** Python 3.10, Scikit-Learn, Pandas, NumPy, Flask, MySQL

---

##  What the Data Taught Me (Feature Importance)
I used a public Kaggle dataset with 50,000 rows of shopper data. When I extracted the feature weights from my model, it showed exactly what makes a customer tick:

1.  **Cart_Abandonment_Rate (38.48%)** — By far the biggest clue. If someone keeps dumping their basket, they are hesitating hard.
2.  **Total_Purchases (18.59%)** — Shows customer loyalty. Returning buyers convert way easier.
3.  **Age (14.28%)** — Good for baseline tracking across generations.
4.  **Pages_Per_Session (11.76%)** — How deep they dig into the site.
5.  **Session_Duration_Avg (10.46%)** — Attention span on the pages.
6.  **Login_Frequency (6.43%)** — How often they open the app in a week.

---

##  How I Built It

### 1. Cleaning the Messy Logs
Real telemetry data is always messy. I wrote a quick preprocessing pipeline using Pandas to find any missing values or empty cells and automatically filled them with the column's mean average so the algorithm wouldn't crash.

### 2. Training the Model
I split the 50,000 rows into an 80/20 train/test split. I chose a **Random Forest Classifier** and tuned it to use 150 decision trees with a max depth of 8 layers. This stopped the code from just memorizing the data and helped it actually learn the patterns.

### 3. Making it Full-Stack
After training, I saved the model weights into a `.pkl` file using Joblib. Then, I wrote a quick **Flask Web Server** script. When a visitor "browses" the site, Flask feeds their stats to the model, and if their cart abandonment risk is too high, it simulates triggering an automatic 15% discount popup banner to save the sale.

---

##  What it Looks Like in the Terminal

```text
 Scanning Live Shopper: Age=42, Session=350s, Cart Abandon=10%
   SYSTEM RESULT: LIKELY TO PURCHASE (Confidence: 86.42%)

 Scanning Live Shopper: Age=23, Session=45s, Cart Abandon=90%
   SYSTEM RESULT: NOT LIKELY TO PURCHASE (Confidence: 78.12%)
   TRIGGERED ACTION: Push 15% discount popup banner to save the sale!
```

---

##  Repository Setup
*   `sample_data/` - Holds the original customer behavior CSV spreadsheet.
*   `ecommerce_behaviour_rf_model.pkl` - The trained brain of the Random Forest model.
*   The main notebook file contains all my cell steps, data visualization plots, and the live web application simulation script.
