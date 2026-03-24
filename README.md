# Cloud-Based Intelligent Sales and Customer Analytics Platform

##  Problem Statement
Businesses generate large volumes of sales and customer data, but often struggle to convert this data into meaningful insights. This project aims to build a cloud-based analytics platform that processes, analyzes, and visualizes business data to support data-driven decision making.

##  Objectives
- Store and process sales and customer data using cloud infrastructure
- Perform data cleaning, analysis, and feature engineering
- Visualize business KPIs using interactive dashboards
- Provide actionable insights for business growth

##  Technologies Used
- Python (Pandas, NumPy)
- AWS (EC2, S3, IAM)
- Power BI / Tableau
- Git & GitHub

##  System Architecture
Raw Data → Cloud Storage (S3) → Processing (EC2 + Python) → Analytics → Dashboard

## Tableau Dashboard

An interactive sales analytics dashboard built in Tableau Public.

**Live Dashboard:** https://public.tableau.com/views/SalesCloudAnalyticsDashboard/SalesDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

### Sheets built:
- KPI overview: total sales, order count, avg order value
- Sales by category and sub-category
- Monthly sales trend with average reference line
- Geographic sales heatmap by US state
- Customer segmentation scatter plot
- Actual vs predicted sales comparison
- Filters: Region, Category, Sub-Category, Year

#Project Structure
sales-cloud-analytics/
│
├── data/                     # Raw & processed datasets
├── scripts/                  # Python scripts (EDA, training, visualization)
├── notebooks/                # Jupyter notebooks (analysis)
├── dashboard/                # Tableau dashboard files
└── README.md

# How to Run the Project
Clone the repository:
git clone https://github.com/shikatare/sales-cloud-analytics.git
cd sales-cloud-analytics
Create a virtual environment:
python3 -m venv venv
source venv/bin/activate
Install dependencies:
pip install pandas numpy scikit-learn matplotlib
Run pipeline:
python scripts/sales_prediction.py




