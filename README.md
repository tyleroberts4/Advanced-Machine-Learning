# Advanced Machine Learning

Course repository for Cal Poly **Advanced Machine Learning**, including homework, in-class work, and the final Rec Center project.

## Featured project: Cal Poly Rec Center usage prediction

Predicts how crowded the Rec Center and individual workout areas will be using three years of Occuspace occupancy data (223k+ observations, 30-minute intervals).

| | |
|---|---|
| **Problem** | Forecast utilization and classify usage as low / medium / high |
| **Best regression** | Stacking ensemble — RMSE 0.123, R² 0.872 |
| **Best classification** | CatBoost — 81.9% accuracy, macro-F1 0.793 |
| **Models** | Ridge, Random Forest, LightGBM, CatBoost, Keras MLP, stacking ensemble |
| **Deliverables** | 4 Jupyter notebooks, business report, Streamlit dashboard |

**Explore the project:** [rec_center/](rec_center/README.md)

- [Final report (PDF)](rec_center/report/Rec_Center_Final_Report.pdf)
- [Final report (Markdown)](rec_center/report/Rec_Center_Final_Report.md)
- [Notebooks](rec_center/notebooks/)

## Repository structure

| Folder | Contents |
|---|---|
| [rec_center/](rec_center/) | Final project — EDA, modeling, report, dashboard |
| [datasets/](datasets/) | Course and project data |
| [Kaggle/](Kaggle/) | Kaggle competition homework |
| [midterm/](midterm/) | Midterm classification project |
| [Homework Notebooks/](Homework%20Notebooks/) | Weekly assignments |
| [In_Class_Activities/](In_Class_Activities/) | In-class exercises |
