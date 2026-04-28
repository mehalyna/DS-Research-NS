"""
Simple Coffee Health Dataset Visualization
==========================================
A streamlined script for visualizing key patterns in the coffee health dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
plt.style.use('default')
sns.set_palette("Set2")

def load_and_explore_data():
    """Load the dataset and show basic information"""
    df = pd.read_csv('synthetic_coffee_health_10000.csv')
    
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def plot_coffee_basics(df):
    """Plot basic coffee consumption patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Coffee Consumption Overview', fontsize=16, fontweight='bold')
    
    # Coffee intake distribution
    axes[0, 0].hist(df['Coffee_Intake'], bins=30, alpha=0.7, color='brown', edgecolor='black')
    axes[0, 0].set_title('Coffee Intake Distribution')
    axes[0, 0].set_xlabel('Cups per day')
    axes[0, 0].set_ylabel('Frequency')
    
    # Caffeine distribution
    axes[0, 1].hist(df['Caffeine_mg'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_title('Caffeine Intake Distribution')
    axes[0, 1].set_xlabel('Caffeine (mg)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Coffee by gender
    coffee_by_gender = df.groupby('Gender')['Coffee_Intake'].mean()
    axes[1, 0].bar(coffee_by_gender.index, coffee_by_gender.values, color=['lightblue', 'lightpink'])
    axes[1, 0].set_title('Average Coffee Intake by Gender')
    axes[1, 0].set_ylabel('Cups per day')
    
    # Coffee by stress level
    stress_order = ['Low', 'Medium', 'High']
    coffee_by_stress = df.groupby('Stress_Level')['Coffee_Intake'].mean().reindex(stress_order)
    axes[1, 1].bar(coffee_by_stress.index, coffee_by_stress.values, color=['green', 'yellow', 'red'])
    axes[1, 1].set_title('Average Coffee Intake by Stress Level')
    axes[1, 1].set_ylabel('Cups per day')
    
    plt.tight_layout()
    plt.show()

def plot_health_metrics(df):
    """Plot key health-related metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Health Metrics Overview', fontsize=16, fontweight='bold')
    
    # Sleep hours distribution
    axes[0, 0].hist(df['Sleep_Hours'], bins=25, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Sleep Hours Distribution')
    axes[0, 0].set_xlabel('Hours per night')
    axes[0, 0].set_ylabel('Frequency')
    
    # BMI distribution
    axes[0, 1].hist(df['BMI'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('BMI Distribution')
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Frequency')
    
    # Heart rate distribution
    axes[1, 0].hist(df['Heart_Rate'], bins=25, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Heart Rate Distribution')
    axes[1, 0].set_xlabel('BPM')
    axes[1, 0].set_ylabel('Frequency')
    
    # Physical activity distribution
    axes[1, 1].hist(df['Physical_Activity_Hours'], bins=25, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Physical Activity Distribution')
    axes[1, 1].set_xlabel('Hours per week')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_coffee_vs_health(df):
    """Explore relationships between coffee consumption and health"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Coffee vs Health Relationships', fontsize=16, fontweight='bold')
    
    # Coffee vs Sleep
    axes[0, 0].scatter(df['Coffee_Intake'], df['Sleep_Hours'], alpha=0.5, color='brown')
    axes[0, 0].set_xlabel('Coffee Intake (cups/day)')
    axes[0, 0].set_ylabel('Sleep Hours')
    axes[0, 0].set_title('Coffee Intake vs Sleep Hours')
    
    # Add correlation coefficient
    corr = df['Coffee_Intake'].corr(df['Sleep_Hours'])
    axes[0, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0, 0].transAxes, 
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Coffee vs Heart Rate
    axes[0, 1].scatter(df['Coffee_Intake'], df['Heart_Rate'], alpha=0.5, color='red')
    axes[0, 1].set_xlabel('Coffee Intake (cups/day)')
    axes[0, 1].set_ylabel('Heart Rate (BPM)')
    axes[0, 1].set_title('Coffee Intake vs Heart Rate')
    
    # Add correlation coefficient
    corr = df['Coffee_Intake'].corr(df['Heart_Rate'])
    axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Coffee by sleep quality
    sleep_quality_order = ['Poor', 'Fair', 'Good', 'Excellent']
    if set(sleep_quality_order).issubset(set(df['Sleep_Quality'].unique())):
        df_copy = df.copy()
        df_copy['Sleep_Quality'] = pd.Categorical(df_copy['Sleep_Quality'], 
                                                categories=sleep_quality_order, ordered=True)
        df_copy.boxplot(column='Coffee_Intake', by='Sleep_Quality', ax=axes[1, 0])
    else:
        df.boxplot(column='Coffee_Intake', by='Sleep_Quality', ax=axes[1, 0])
    
    axes[1, 0].set_title('Coffee Intake by Sleep Quality')
    axes[1, 0].set_xlabel('Sleep Quality')
    axes[1, 0].set_ylabel('Coffee Intake (cups/day)')
    
    # Coffee by health issues
    df.boxplot(column='Coffee_Intake', by='Health_Issues', ax=axes[1, 1])
    axes[1, 1].set_title('Coffee Intake by Health Issues')
    axes[1, 1].set_xlabel('Health Issues')
    axes[1, 1].set_ylabel('Coffee Intake (cups/day)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_demographics(df):
    """Plot demographic information"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Demographics and Lifestyle', fontsize=16, fontweight='bold')
    
    # Age distribution
    axes[0, 0].hist(df['Age'], bins=25, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # Top 10 countries
    top_countries = df['Country'].value_counts().head(10)
    axes[0, 1].barh(range(len(top_countries)), top_countries.values, color='lightgreen')
    axes[0, 1].set_yticks(range(len(top_countries)))
    axes[0, 1].set_yticklabels(top_countries.index)
    axes[0, 1].set_title('Top 10 Countries by Sample Size')
    axes[0, 1].set_xlabel('Count')
    
    # Occupation distribution
    occupation_counts = df['Occupation'].value_counts()
    axes[1, 0].pie(occupation_counts.values, labels=occupation_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Occupation Distribution')
    
    # Smoking and alcohol
    lifestyle_data = pd.DataFrame({
        'Smoking': df['Smoking'].value_counts(),
        'Alcohol': df['Alcohol_Consumption'].value_counts()
    })
    lifestyle_data.plot(kind='bar', ax=axes[1, 1], color=['lightcoral', 'lightblue'])
    axes[1, 1].set_title('Smoking and Alcohol Consumption')
    axes[1, 1].set_xlabel('Yes (1) / No (0)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()

def create_correlation_heatmap(df):
    """Create a correlation heatmap for numerical variables"""
    # Select numerical columns
    numerical_cols = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 'BMI', 
                     'Heart_Rate', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption']
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('Correlation Matrix - Coffee Health Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def print_key_insights(df):
    """Print key insights from the data"""
    print("\n" + "="*50)
    print("KEY INSIGHTS FROM THE COFFEE HEALTH DATASET")
    print("="*50)
    
    print(f"\n📊 Sample Size: {len(df):,} participants")
    print(f"🌍 Countries: {df['Country'].nunique()} different countries")
    print(f"👥 Age Range: {df['Age'].min()} - {df['Age'].max()} years")
    
    print(f"\n☕ Coffee Consumption:")
    print(f"   Average: {df['Coffee_Intake'].mean():.2f} cups/day")
    print(f"   Range: {df['Coffee_Intake'].min():.1f} - {df['Coffee_Intake'].max():.1f} cups/day")
    print(f"   Average Caffeine: {df['Caffeine_mg'].mean():.1f} mg/day")
    
    print(f"\n💤 Sleep Patterns:")
    print(f"   Average Sleep: {df['Sleep_Hours'].mean():.1f} hours/night")
    print(f"   Most Common Sleep Quality: {df['Sleep_Quality'].mode().iloc[0]}")
    
    print(f"\n💪 Health Metrics:")
    print(f"   Average BMI: {df['BMI'].mean():.1f}")
    print(f"   Average Heart Rate: {df['Heart_Rate'].mean():.1f} BPM")
    print(f"   Average Physical Activity: {df['Physical_Activity_Hours'].mean():.1f} hours/week")
    
    # Key correlations
    print(f"\n🔗 Key Correlations:")
    corr_coffee_sleep = df['Coffee_Intake'].corr(df['Sleep_Hours'])
    corr_coffee_heart = df['Coffee_Intake'].corr(df['Heart_Rate'])
    corr_activity_bmi = df['Physical_Activity_Hours'].corr(df['BMI'])
    
    print(f"   Coffee ↔ Sleep Hours: {corr_coffee_sleep:.3f}")
    print(f"   Coffee ↔ Heart Rate: {corr_coffee_heart:.3f}")
    print(f"   Physical Activity ↔ BMI: {corr_activity_bmi:.3f}")
    
    # Health issues breakdown
    print(f"\n🏥 Health Issues Distribution:")
    health_dist = df['Health_Issues'].value_counts(normalize=True) * 100
    for health, pct in health_dist.items():
        print(f"   {health}: {pct:.1f}%")

def main():
    """Main function to run all visualizations"""
    print("🔍 Coffee Health Dataset Visualization")
    print("=" * 40)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Generate all plots
    print("\n📈 Generating visualizations...")
    plot_coffee_basics(df)
    plot_health_metrics(df)
    plot_coffee_vs_health(df)
    plot_demographics(df)
    create_correlation_heatmap(df)
    
    # Print insights
    print_key_insights(df)
    
    print("\n✅ Visualization complete!")
    print("💡 Check the generated plots to explore coffee consumption patterns and health relationships.")

if __name__ == "__main__":
    main()