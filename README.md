# Titanic Survival Analysis — EDA & Feature Engineering

This was my first proper data science project. I wanted to go beyond just loading a dataset and running `.describe()` — I wanted to actually *understand* the data before throwing it at a model. So this notebook is entirely focused on Exploratory Data Analysis and Feature Engineering, no ML yet.

---

## What This Project Is About

The Titanic dataset is one of the most popular beginner datasets on Kaggle, but most people rush straight to building a classifier. I took a step back and asked a more interesting question:

**Was survival on the Titanic random, or did things like wealth, gender, and family structure decide who got a seat on a lifeboat?**

Spoiler: it was very much not random.

---

## Dataset

- Source: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- Training set: 891 passengers, 12 columns
- Target variable: `Survived` (0 = did not survive, 1 = survived)

---

## Project Structure

```
titanic-eda/
│
├── Titanic_EDA.ipynb       # Main notebook (all analysis here)
├── train.csv               # Original training data
├── test.csv                # Original test data
├── Titanic_modified.csv    # Cleaned + feature-engineered dataset
└── README.md
```

---

## What I Did (Step by Step)

### 1. Data Understanding
Before writing a single line of analysis, I built a proper data dictionary — what each column means, what type it is, and what problems it has (missing values, wrong interpretations, etc.). For example, I noticed that the `Fare` column stores the *group* fare, not the individual fare. That's the kind of thing you miss if you just skim the data.

### 2. Univariate Analysis
Looked at each column individually — distributions, skewness, outliers, missing value percentages. Key things I found:
- `Age` is roughly normally distributed but has ~20% missing values
- `Fare` is heavily right-skewed (a few passengers paid way more than everyone else)
- `Cabin` is 77% missing, which makes it tricky to use directly

### 3. Handling Missing Values
I didn't just fill everything with the global mean. I filled `Age` using the **mean age per passenger class** — because a 1st class passenger is likely older than a 3rd class passenger on average, so using one global mean would be inaccurate. For `Cabin`, since 77% was missing, I assigned a dummy value `'M'` so I could still extract the deck letter from the ones that existed.

### 4. Feature Engineering
This is the part I enjoyed most. I created 8 new features from the raw columns:

| Feature | How I Made It | Why |
|---|---|---|
| `individual_fare` | Fare ÷ group size | The original fare was for the whole group, not per person |
| `family_size` | SibSp + Parch + 1 | Combines siblings/spouses and parents/children into one number |
| `family_type` | Categorised family_size | Alone / Small (2–4) / Large (5+) — survival patterns differ by group type |
| `Title` | Extracted from Name column | Captures gender + social class in one variable (Mr, Mrs, Miss, Master, etc.) |
| `Deck` | First letter of Cabin | The deck letter tells you how close you were to the lifeboats |
| `Ticket_frequency` | How many people share a ticket | Actual group size, not just family — captures friends/colleagues too |
| `Ticket_prefix` | Prefix from ticket number | Ticket codes like `PC`, `CA`, `SOTON` correlate strongly with class |
| `non_family_group` | Ticket frequency > family size | Identifies passengers who were in a group but not related by family |

The ticket features were probably the most interesting discovery. SibSp and Parch only capture *family* — but the ticket data shows that many passengers shared a ticket with non-family members (friends, colleagues, travel groups). That's information the original columns completely missed.

### 5. Bivariate Analysis
Looked at how each feature relates to survival. Some clear patterns:
- Higher class → higher survival
- Female → significantly higher survival
- Passengers from Cherbourg (C) seemed to survive more — I later figured out why (see multivariate section)

### 6. Multivariate Analysis
This is where things got more interesting. Looking at one variable at a time gives you a limited picture. Some findings:
- **97.6% of 1st class women survived vs ~50% of 3rd class women** — the "women first" protocol existed, but class determined how well it was enforced
- **The Cherbourg mystery**: Passengers from Cherbourg had a higher survival rate — but not because of where they boarded. It turned out ~50% of Cherbourg passengers were 1st class and a higher proportion were female. Classic example of what's called **Simpson's Paradox**
- **Children in 3rd class still had worse survival** than children in higher classes, likely because 3rd class cabins were deep in the ship and farther from lifeboats

---

## Key Findings

1. Class was a matter of life and death. 1st class survival rate: ~63%. 3rd class: ~24%.
2. Gender protected you, but class decided *how much* that protection was worth.
3. Small families (2–4 people) had better survival odds than solo travellers or large families.
4. The ticket prefix and frequency features reveal social structures that the original columns don't capture.
5. Deck position (extracted from Cabin) echoed passenger class and correlated with survival.

---

## Tools Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## How to Run

1. Clone this repo
2. Make sure you have the required libraries: `pip install pandas numpy matplotlib seaborn`
3. Place `train.csv` and `test.csv` in the same folder as the notebook
4. Open `Titanic_EDA.ipynb` and run all cells

---

## What's Next

This notebook only covers EDA and Feature Engineering. The cleaned dataset (`Titanic_modified.csv`) is ready to be used for modelling. Next step would be building a classifier (probably starting with Logistic Regression, then Random Forest) using the engineered features.

---

## Honest Limitations

- 77% of Cabin data is missing, so Deck-based conclusions are directional, not definitive
- Some group combinations (like large families in 2nd class) have very small sample sizes, so their survival percentages can be misleading
- All survival analysis is based on the 891-row training set, not the full 1309 passengers

---

*Made by Shahbar Hussain | First DS Project | Dataset from Kaggle*
