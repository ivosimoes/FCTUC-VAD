# FCTUC-VAD 2024/2025

Project made for 2024/2025 edition of Advanced Data Visualization course

## Summary

**Theme**: Airbnbs in Porto: A Data Visualization Journey to Support Hosts and City Administration Decisions 

**Objective**: This project focuses on the development of a dashboard containing a variety of visualizations to address a wide array of questions regarding Airbnbs in the city of Porto, Portugal. Based on the specified questions and design requirements, the work tackles the various aspects of the data visualization pipeline, from the acquisition of data to the evaluation and critical reflection of the developed dashboard.

## Group
- António Marques (uc2022259825)
- Ivo Simões (uc2024160048)

## Repository Structure

```text
├── data                    # Preprocessed datasets utilized in EDA and the visualizations
├── assets                  # Icons for the map of Airbnb listings
├── EDA.ipynb               # Notebook containing exploratory data analysis procedures
├── requirements.txt        # List of core Python libraries utilized in this project
├── VAD_Report.pdf          # Report detailing the problem, design decisions, implementation, and comments
├── website.py              # Source code to host the dashboard's website
└── README.md               # Project details
```

## Execution Setup and Side Notes

### Install requirements

The project was made and tested using Python 3.12.9. To install the necessary packages to run the project, do **pip install -r requirements.txt**.

### Data

**Make sure** that the *assets* and *data* folders, containing the necessary sprites and datasets, respectively, are in the same directory as the *website.py* script.

*IMPORTANT: The dataset pertaining to the reviews (reviews_complete_final_v2.csv) is hosted on [HuggingFace](https://huggingface.co/datasets/ivosimoes/FCTUC-VAD-Reviews) due to file size.*

### Running and accessing the website

Execute the *website.py* script (**python website.py**). By default, the website will be running locally on **127.0.0.1:8050** (you can access it on http://127.0.0.1:8050). You can change the host settings on the very last line of the source file.

### Extra

The *EDA.ipynb* notebook contains some of the exploratory data analysis used to make decisions regarding the visualizations present in the website, as well as the transformations that were applied to the data in order to make the visualizations functional and efficient. The notebook also contains some comments giving insight into the transformations.
