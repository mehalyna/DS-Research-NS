# ☕ Coffee Health Dataset Analysis

A comprehensive analysis and visualization toolkit for exploring the relationship between coffee consumption and health metrics using a synthetic dataset of 10,000 participants.

## 📊 Dataset Overview

The `synthetic_coffee_health_10000.csv` dataset contains detailed information about coffee consumption habits and health metrics across diverse demographics. This synthetic dataset was created to explore potential relationships between caffeine intake and various health indicators.

### Dataset Features

**Demographics:**
- `ID` - Unique participant identifier
- `Age` - Participant age (18-65+ years)
- `Gender` - Male/Female
- `Country` - Country of residence
- `Occupation` - Professional category

**Coffee Consumption:**
- `Coffee_Intake` - Daily coffee consumption (cups per day)
- `Caffeine_mg` - Daily caffeine intake (milligrams)

**Health Metrics:**
- `Sleep_Hours` - Average nightly sleep duration
- `Sleep_Quality` - Self-reported sleep quality (Poor/Fair/Good/Excellent)
- `BMI` - Body Mass Index
- `Heart_Rate` - Resting heart rate (BPM)
- `Stress_Level` - Self-reported stress (Low/Medium/High)
- `Physical_Activity_Hours` - Weekly physical activity hours
- `Health_Issues` - Current health status (None/Mild/Moderate/Severe)

**Lifestyle Factors:**
- `Smoking` - Smoking status (0=No, 1=Yes)
- `Alcohol_Consumption` - Alcohol consumption (0=No, 1=Yes)

## 🎯 Key Research Questions

This dataset enables exploration of several interesting questions:

1. **Does coffee consumption affect sleep patterns?**
2. **Is there a relationship between caffeine intake and heart rate?**
3. **How does stress level correlate with coffee consumption?**
4. **Are there demographic differences in coffee consumption habits?**
5. **What's the relationship between physical activity and health metrics?**

## 🛠️ Tools and Files

### Visualization Scripts

1. **`coffee_visualizer.py`** - *Recommended for beginners*
   - Streamlined visualization script
   - Easy to run and understand
   - Comprehensive overview of key patterns
   - Automated insights generation

2. **`visualize_coffee_health.py`** - *Advanced analysis*
   - Comprehensive visualization suite
   - Advanced statistical analysis
   - Multi-variable relationship exploration
   - Detailed correlation analysis

3. **`coffee_health_analysis.ipynb`** - *Interactive exploration*
   - Jupyter notebook for step-by-step analysis
   - Interactive plotting capabilities
   - Space for custom analysis
   - Educational format with explanations

### Requirements

All required Python packages are listed in `requirements.txt`:
- pandas (data manipulation)
- matplotlib (plotting)
- seaborn (statistical visualization)
- numpy (numerical computations)

## 🚀 Quick Start

### Option 1: Python Script (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main visualizer:**
   ```bash
   python coffee_visualizer.py
   ```

This will generate all visualizations and print key insights to the console.

### Option 2: Jupyter Notebook (Interactive)

1. **Install Jupyter (if not already installed):**
   ```bash
   pip install jupyter
   ```

2. **Launch the notebook:**
   ```bash
   jupyter notebook coffee_health_analysis.ipynb
   ```

3. **Run cells sequentially** to explore the data interactively.

### Option 3: Advanced Analysis

For comprehensive analysis with additional visualizations:
```bash
python visualize_coffee_health.py
```

## 📈 What You'll Discover

### Generated Visualizations

1. **Distribution Analysis**
   - Coffee intake and caffeine consumption patterns
   - Health metrics distributions (sleep, BMI, heart rate)
   - Demographic breakdowns

2. **Relationship Analysis**
   - Coffee vs. sleep quality correlations
   - Caffeine vs. heart rate relationships
   - Stress level vs. coffee consumption

3. **Correlation Matrix**
   - Comprehensive correlation heatmap
   - Identification of strongest relationships
   - Statistical significance indicators

4. **Demographic Insights**
   - Country-wise coffee consumption patterns
   - Gender differences in health metrics
   - Age-related trends

### Sample Insights

The analysis typically reveals patterns such as:
- Average coffee consumption: ~3.2 cups/day
- Average caffeine intake: ~305mg/day
- Correlation between coffee and sleep: typically negative
- Relationship between stress and coffee consumption
- Geographic variations in consumption patterns

## 🔍 Customization

### Modifying the Analysis

You can easily customize the analysis by:

1. **Adding new visualizations** to any of the Python scripts
2. **Filtering data** by specific demographics or health conditions
3. **Creating custom correlation analyses** for variables of interest
4. **Adjusting visualization parameters** (colors, chart types, etc.)

### Example Custom Analysis

```python
# Filter data for specific analysis
high_coffee_drinkers = df[df['Coffee_Intake'] > 4]
low_sleep_participants = df[df['Sleep_Hours'] < 6]

# Create custom visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Coffee_Intake'], 
           c=df['Stress_Level'].map({'Low': 'green', 'Medium': 'orange', 'High': 'red'}))
plt.xlabel('Age')
plt.ylabel('Coffee Intake')
plt.title('Coffee Consumption vs Age by Stress Level')
plt.legend()
plt.show()
```

## 📋 Data Quality

This synthetic dataset is designed for:
- ✅ Educational purposes
- ✅ Visualization practice
- ✅ Statistical analysis learning
- ✅ Data science methodology demonstration

**Note:** This is synthetic data created for analysis practice. Results should not be used for actual health decisions or research conclusions.

## 🤝 Contributing

Feel free to:
- Add new visualization types
- Improve existing analysis methods
- Create additional insights
- Enhance documentation
- Report issues or suggestions

## 📚 Additional Resources

### Learning More About the Data

- **Statistical Analysis**: The correlation matrix reveals key relationships
- **Data Visualization**: Multiple chart types showcase different aspects
- **Health Insights**: Patterns may reflect real-world coffee consumption trends

### Extending the Analysis

Consider exploring:
- Machine learning prediction models
- Time-series analysis (if temporal data were available)
- Clustering analysis for participant segmentation
- Advanced statistical testing for significance

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed with `pip install -r requirements.txt`
2. **File Not Found**: Make sure you're running scripts from the correct directory
3. **Display Issues**: For notebooks, ensure `%matplotlib inline` is executed
4. **Memory Issues**: If dataset is large, consider sampling for initial exploration

### Getting Help

- Check that all required files are in the same directory
- Verify Python version compatibility (3.7+ recommended)
- Ensure matplotlib backend is properly configured for your system

## 📄 License

This project is provided for educational and research purposes. The synthetic dataset is freely available for analysis and learning.

---

**Happy analyzing! ☕📊**

*Explore the fascinating relationships between coffee consumption and health metrics through data visualization and statistical analysis.*