"""
Coffee Health Dataset Visualization
===================================
This script creates comprehensive visualizations for the synthetic coffee health dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv('synthetic_coffee_health_10000.csv')
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("Dataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nBasic Statistics:")
    print(df.describe())

def create_distribution_plots(df):
    """Create distribution plots for key numerical variables"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Key Health and Coffee Variables', fontsize=16, fontweight='bold')
    
    # Coffee Intake Distribution
    axes[0, 0].hist(df['Coffee_Intake'], bins=30, alpha=0.7, color='brown')
    axes[0, 0].set_title('Coffee Intake Distribution (cups/day)')
    axes[0, 0].set_xlabel('Coffee Intake')
    axes[0, 0].set_ylabel('Frequency')
    
    # Caffeine Distribution
    axes[0, 1].hist(df['Caffeine_mg'], bins=30, alpha=0.7, color='orange')
    axes[0, 1].set_title('Caffeine Intake Distribution (mg)')
    axes[0, 1].set_xlabel('Caffeine (mg)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Sleep Hours Distribution
    axes[0, 2].hist(df['Sleep_Hours'], bins=20, alpha=0.7, color='blue')
    axes[0, 2].set_title('Sleep Hours Distribution')
    axes[0, 2].set_xlabel('Sleep Hours')
    axes[0, 2].set_ylabel('Frequency')
    
    # BMI Distribution
    axes[1, 0].hist(df['BMI'], bins=30, alpha=0.7, color='green')
    axes[1, 0].set_title('BMI Distribution')
    axes[1, 0].set_xlabel('BMI')
    axes[1, 0].set_ylabel('Frequency')
    
    # Heart Rate Distribution
    axes[1, 1].hist(df['Heart_Rate'], bins=25, alpha=0.7, color='red')
    axes[1, 1].set_title('Heart Rate Distribution (bpm)')
    axes[1, 1].set_xlabel('Heart Rate (bpm)')
    axes[1, 1].set_ylabel('Frequency')
    
    # Physical Activity Distribution
    axes[1, 2].hist(df['Physical_Activity_Hours'], bins=25, alpha=0.7, color='purple')
    axes[1, 2].set_title('Physical Activity Distribution (hours/week)')
    axes[1, 2].set_xlabel('Physical Activity (hours/week)')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def create_categorical_plots(df):
    """Create plots for categorical variables"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Categorical Variables', fontsize=16, fontweight='bold')
    
    # Gender Distribution
    gender_counts = df['Gender'].value_counts()
    axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Gender Distribution')
    
    # Sleep Quality Distribution
    sleep_quality_counts = df['Sleep_Quality'].value_counts()
    axes[0, 1].bar(sleep_quality_counts.index, sleep_quality_counts.values, color='lightblue')
    axes[0, 1].set_title('Sleep Quality Distribution')
    axes[0, 1].set_xlabel('Sleep Quality')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Stress Level Distribution
    stress_counts = df['Stress_Level'].value_counts()
    axes[0, 2].bar(stress_counts.index, stress_counts.values, color='lightcoral')
    axes[0, 2].set_title('Stress Level Distribution')
    axes[0, 2].set_xlabel('Stress Level')
    axes[0, 2].set_ylabel('Count')
    
    # Health Issues Distribution
    health_counts = df['Health_Issues'].value_counts()
    axes[1, 0].pie(health_counts.values, labels=health_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Health Issues Distribution')
    
    # Top 10 Countries
    country_counts = df['Country'].value_counts().head(10)
    axes[1, 1].barh(country_counts.index, country_counts.values, color='lightgreen')
    axes[1, 1].set_title('Top 10 Countries by Sample Size')
    axes[1, 1].set_xlabel('Count')
    
    # Occupation Distribution
    occupation_counts = df['Occupation'].value_counts()
    axes[1, 2].pie(occupation_counts.values, labels=occupation_counts.index, autopct='%1.1f%%')
    axes[1, 2].set_title('Occupation Distribution')
    
    plt.tight_layout()
    plt.show()

def create_correlation_analysis(df):
    """Create correlation heatmap and analysis"""
    # Select numerical columns for correlation
    numerical_cols = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 'BMI', 
                     'Heart_Rate', 'Physical_Activity_Hours', 'Smoking', 'Alcohol_Consumption']
    
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_coffee_health_relationships(df):
    """Analyze relationships between coffee consumption and health metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Coffee Consumption vs Health Metrics', fontsize=16, fontweight='bold')
    
    # Coffee Intake vs Sleep Hours
    axes[0, 0].scatter(df['Coffee_Intake'], df['Sleep_Hours'], alpha=0.5, color='brown')
    axes[0, 0].set_xlabel('Coffee Intake (cups/day)')
    axes[0, 0].set_ylabel('Sleep Hours')
    axes[0, 0].set_title('Coffee Intake vs Sleep Hours')
    
    # Add trend line
    z = np.polyfit(df['Coffee_Intake'], df['Sleep_Hours'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df['Coffee_Intake'], p(df['Coffee_Intake']), "r--", alpha=0.8)
    
    # Coffee Intake vs Heart Rate
    axes[0, 1].scatter(df['Coffee_Intake'], df['Heart_Rate'], alpha=0.5, color='red')
    axes[0, 1].set_xlabel('Coffee Intake (cups/day)')
    axes[0, 1].set_ylabel('Heart Rate (bpm)')
    axes[0, 1].set_title('Coffee Intake vs Heart Rate')
    
    # Add trend line
    z = np.polyfit(df['Coffee_Intake'], df['Heart_Rate'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['Coffee_Intake'], p(df['Coffee_Intake']), "r--", alpha=0.8)
    
    # Coffee Intake vs BMI
    axes[1, 0].scatter(df['Coffee_Intake'], df['BMI'], alpha=0.5, color='green')
    axes[1, 0].set_xlabel('Coffee Intake (cups/day)')
    axes[1, 0].set_ylabel('BMI')
    axes[1, 0].set_title('Coffee Intake vs BMI')
    
    # Add trend line
    z = np.polyfit(df['Coffee_Intake'], df['BMI'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(df['Coffee_Intake'], p(df['Coffee_Intake']), "r--", alpha=0.8)
    
    # Coffee Intake by Stress Level (Box plot)
    df.boxplot(column='Coffee_Intake', by='Stress_Level', ax=axes[1, 1])
    axes[1, 1].set_title('Coffee Intake by Stress Level')
    axes[1, 1].set_xlabel('Stress Level')
    axes[1, 1].set_ylabel('Coffee Intake (cups/day)')
    
    plt.tight_layout()
    plt.show()

def create_advanced_visualizations(df):
    """Create advanced visualizations with multiple variables"""
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 3, figure=fig)
    
    # Age vs Coffee Intake colored by Gender
    ax1 = fig.add_subplot(gs[0, 0])
    for gender in df['Gender'].unique():
        gender_data = df[df['Gender'] == gender]
        ax1.scatter(gender_data['Age'], gender_data['Coffee_Intake'], 
                   label=gender, alpha=0.6, s=30)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Coffee Intake (cups/day)')
    ax1.set_title('Age vs Coffee Intake by Gender')
    ax1.legend()
    
    # Sleep Quality vs Sleep Hours
    ax2 = fig.add_subplot(gs[0, 1])
    sleep_quality_order = ['Poor', 'Fair', 'Good', 'Excellent']
    df_sorted = df.copy()
    df_sorted['Sleep_Quality'] = pd.Categorical(df_sorted['Sleep_Quality'], 
                                               categories=sleep_quality_order, 
                                               ordered=True)
    df_sorted.boxplot(column='Sleep_Hours', by='Sleep_Quality', ax=ax2)
    ax2.set_title('Sleep Hours by Sleep Quality')
    ax2.set_xlabel('Sleep Quality')
    ax2.set_ylabel('Sleep Hours')
    
    # Physical Activity vs BMI colored by Health Issues
    ax3 = fig.add_subplot(gs[0, 2])
    health_colors = {'None': 'green', 'Mild': 'orange', 'Moderate': 'red', 'Severe': 'darkred'}
    for health in df['Health_Issues'].unique():
        health_data = df[df['Health_Issues'] == health]
        ax3.scatter(health_data['Physical_Activity_Hours'], health_data['BMI'], 
                   label=health, alpha=0.6, s=30, color=health_colors.get(health, 'gray'))
    ax3.set_xlabel('Physical Activity (hours/week)')
    ax3.set_ylabel('BMI')
    ax3.set_title('Physical Activity vs BMI by Health Issues')
    ax3.legend()
    
    # Coffee Intake distribution by Country (top 5)
    ax4 = fig.add_subplot(gs[1, :])
    top_countries = df['Country'].value_counts().head(5).index
    df_top_countries = df[df['Country'].isin(top_countries)]
    
    coffee_by_country = []
    countries = []
    for country in top_countries:
        coffee_by_country.append(df_top_countries[df_top_countries['Country'] == country]['Coffee_Intake'].values)
        countries.append(country)
    
    ax4.boxplot(coffee_by_country, labels=countries)
    ax4.set_title('Coffee Intake Distribution by Top 5 Countries')
    ax4.set_xlabel('Country')
    ax4.set_ylabel('Coffee Intake (cups/day)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Caffeine vs Heart Rate with size by Age
    ax5 = fig.add_subplot(gs[2, 0])
    sizes = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min()) * 100 + 10
    scatter = ax5.scatter(df['Caffeine_mg'], df['Heart_Rate'], s=sizes, alpha=0.6, c=df['Age'], cmap='viridis')
    ax5.set_xlabel('Caffeine (mg)')
    ax5.set_ylabel('Heart Rate (bpm)')
    ax5.set_title('Caffeine vs Heart Rate (bubble size = Age)')
    plt.colorbar(scatter, ax=ax5, label='Age')
    
    # Stress Level vs Sleep Quality heatmap
    ax6 = fig.add_subplot(gs[2, 1])
    stress_sleep_crosstab = pd.crosstab(df['Stress_Level'], df['Sleep_Quality'])
    sns.heatmap(stress_sleep_crosstab, annot=True, fmt='d', cmap='Blues', ax=ax6)
    ax6.set_title('Stress Level vs Sleep Quality')
    ax6.set_xlabel('Sleep Quality')
    ax6.set_ylabel('Stress Level')
    
    # Age distribution by occupation
    ax7 = fig.add_subplot(gs[2, 2])
    age_by_occupation = []
    occupations = []
    for occupation in df['Occupation'].unique():
        age_by_occupation.append(df[df['Occupation'] == occupation]['Age'].values)
        occupations.append(occupation)
    
    ax7.boxplot(age_by_occupation, labels=occupations)
    ax7.set_title('Age Distribution by Occupation')
    ax7.set_xlabel('Occupation')
    ax7.set_ylabel('Age')
    ax7.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_summary_stats(df):
    """Create summary statistics and insights"""
    print("\n" + "="*60)
    print("COFFEE HEALTH DATASET SUMMARY INSIGHTS")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Variables: {len(df.columns)}")
    print(f"   • Countries represented: {df['Country'].nunique()}")
    
    print(f"\n☕ Coffee Consumption:")
    print(f"   • Average coffee intake: {df['Coffee_Intake'].mean():.2f} cups/day")
    print(f"   • Average caffeine intake: {df['Caffeine_mg'].mean():.1f} mg/day")
    print(f"   • Max coffee intake: {df['Coffee_Intake'].max():.1f} cups/day")
    
    print(f"\n💤 Sleep Patterns:")
    print(f"   • Average sleep: {df['Sleep_Hours'].mean():.1f} hours/night")
    print(f"   • Most common sleep quality: {df['Sleep_Quality'].mode().iloc[0]}")
    
    print(f"\n💪 Health Metrics:")
    print(f"   • Average BMI: {df['BMI'].mean():.1f}")
    print(f"   • Average heart rate: {df['Heart_Rate'].mean():.1f} bpm")
    print(f"   • Average physical activity: {df['Physical_Activity_Hours'].mean():.1f} hours/week")
    
    print(f"\n🎯 Key Correlations:")
    corr_coffee_sleep = df['Coffee_Intake'].corr(df['Sleep_Hours'])
    corr_coffee_heart = df['Coffee_Intake'].corr(df['Heart_Rate'])
    corr_activity_bmi = df['Physical_Activity_Hours'].corr(df['BMI'])
    
    print(f"   • Coffee vs Sleep: {corr_coffee_sleep:.3f}")
    print(f"   • Coffee vs Heart Rate: {corr_coffee_heart:.3f}")
    print(f"   • Physical Activity vs BMI: {corr_activity_bmi:.3f}")
    
    print(f"\n🌍 Demographics:")
    print(f"   • Most represented country: {df['Country'].mode().iloc[0]}")
    print(f"   • Gender split: {(df['Gender'].value_counts(normalize=True) * 100).round(1).to_dict()}")
    print(f"   • Age range: {df['Age'].min()}-{df['Age'].max()} years")

def main():
    """Main function to run all visualizations"""
    print("Loading Coffee Health Dataset...")
    df = load_data()
    
    print("Generating visualizations...")
    
    # Basic dataset information
    basic_info(df)
    
    # Create all visualizations
    create_distribution_plots(df)
    create_categorical_plots(df)
    create_correlation_analysis(df)
    create_coffee_health_relationships(df)
    create_advanced_visualizations(df)
    
    # Summary statistics
    create_summary_stats(df)
    
    print("\n✅ All visualizations have been generated successfully!")
    print("🔍 Check the plots above for insights into coffee consumption and health relationships.")

if __name__ == "__main__":
    main()