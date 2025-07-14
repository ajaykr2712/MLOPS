#!/usr/bin/env python
"""
Sample data generator for MDT Dashboard.
Creates realistic datasets for testing and demonstration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime, timedelta

def create_customer_churn_dataset(n_samples=10000, drift_factor=0.0):
    """Create a realistic customer churn dataset."""
    np.random.seed(42 + int(drift_factor * 100))
    
    # Customer demographics
    age = np.random.normal(35 + drift_factor * 5, 12, n_samples).clip(18, 80)
    income = np.random.lognormal(10.8 + drift_factor * 0.2, 0.5, n_samples).clip(20000, 200000)
    
    # Service usage patterns
    tenure_months = np.random.exponential(24 + drift_factor * 6, n_samples).clip(1, 120)
    monthly_usage_gb = np.random.gamma(2, 5 + drift_factor * 2, n_samples).clip(0.1, 100)
    calls_per_month = np.random.poisson(30 + drift_factor * 10, n_samples).clip(0, 200)
    
    # Service quality metrics
    support_tickets = np.random.poisson(2 + drift_factor * 1, n_samples).clip(0, 20)
    network_quality = np.random.beta(8, 2, n_samples) + drift_factor * 0.1
    payment_delays = np.random.poisson(1 + drift_factor * 2, n_samples).clip(0, 30)
    
    # Behavioral features
    app_downloads = np.random.poisson(5 + drift_factor * 3, n_samples).clip(0, 50)
    social_media_usage = np.random.exponential(2 + drift_factor, n_samples).clip(0, 20)
    roaming_usage = np.random.exponential(0.5 + drift_factor * 0.5, n_samples).clip(0, 10)
    
    # Contract features
    contract_types = np.random.choice(['monthly', 'annual', 'biennial'], n_samples, p=[0.6, 0.3, 0.1])
    plan_types = np.random.choice(['basic', 'premium', 'enterprise'], n_samples, p=[0.5, 0.3, 0.2])
    auto_pay = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Geographic features
    regions = np.random.choice(['north', 'south', 'east', 'west'], n_samples, p=[0.3, 0.25, 0.25, 0.2])
    urban = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Calculate churn probability based on features
    churn_prob = (
        0.05 +  # base churn rate
        (age < 25) * 0.15 +  # young customers churn more
        (age > 65) * 0.1 +   # elderly customers churn more
        (income < 30000) * 0.2 +  # low income churn more
        (tenure_months < 6) * 0.3 +  # new customers churn more
        (support_tickets > 5) * 0.25 +  # high support needs
        (payment_delays > 3) * 0.2 +  # payment issues
        (network_quality < 0.5) * 0.15 +  # poor quality
        (contract_types == 'monthly') * 0.1 +  # monthly contracts churn more
        np.random.normal(0, 0.05, n_samples)  # noise
    )
    
    # Add drift effect
    if drift_factor > 0:
        # Simulate economic downturn affecting churn patterns
        economic_pressure = (income < 50000) * drift_factor * 0.3
        churn_prob += economic_pressure
    
    churn_prob = np.clip(churn_prob, 0, 0.8)
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        # Demographics
        'age': age.round().astype(int),
        'income': income.round(2),
        'region': regions,
        'urban': urban,
        
        # Service usage
        'tenure_months': tenure_months.round().astype(int),
        'monthly_usage_gb': monthly_usage_gb.round(2),
        'calls_per_month': calls_per_month,
        'app_downloads': app_downloads,
        'social_media_hours': social_media_usage.round(2),
        'roaming_charges': roaming_usage.round(2),
        
        # Service quality
        'support_tickets': support_tickets,
        'network_quality_score': (network_quality * 10).round(2),
        'payment_delay_days': payment_delays,
        
        # Contract
        'contract_type': contract_types,
        'plan_type': plan_types,
        'auto_pay': auto_pay,
        
        # Target
        'churn': churn
    })
    
    return df

def create_credit_scoring_dataset(n_samples=8000, drift_factor=0.0):
    """Create a realistic credit scoring dataset."""
    np.random.seed(123 + int(drift_factor * 100))
    
    # Personal information
    age = np.random.normal(40 + drift_factor * 3, 15, n_samples).clip(18, 85)
    income = np.random.lognormal(10.5 + drift_factor * 0.1, 0.6, n_samples).clip(15000, 300000)
    employment_length = np.random.exponential(8 + drift_factor * 2, n_samples).clip(0, 40)
    
    # Financial history
    existing_loans = np.random.poisson(2 + drift_factor, n_samples).clip(0, 15)
    credit_history_length = np.random.gamma(3, 4 + drift_factor, n_samples).clip(0, 30)
    debt_to_income = np.random.beta(2, 5, n_samples) + drift_factor * 0.1
    
    # Credit behavior
    credit_utilization = np.random.beta(2, 3, n_samples) + drift_factor * 0.15
    payment_history_score = np.random.beta(8, 2, n_samples) - drift_factor * 0.1
    inquiries_last_6m = np.random.poisson(1 + drift_factor * 2, n_samples).clip(0, 20)
    
    # Loan specifics
    loan_amount = np.random.lognormal(9.5, 0.8, n_samples).clip(1000, 100000)
    loan_purpose = np.random.choice(['home', 'auto', 'personal', 'business'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    collateral_value = loan_amount * np.random.uniform(0.8, 1.5, n_samples)
    
    # Calculate default probability
    default_prob = (
        0.03 +  # base default rate
        (age < 25) * 0.05 +  # young borrowers higher risk
        (income < 25000) * 0.15 +  # low income higher risk
        (employment_length < 2) * 0.1 +  # unstable employment
        (debt_to_income > 0.5) * 0.2 +  # high debt ratio
        (credit_utilization > 0.8) * 0.15 +  # high utilization
        (payment_history_score < 0.7) * 0.25 +  # poor payment history
        (inquiries_last_6m > 5) * 0.1 +  # credit seeking behavior
        (existing_loans > 5) * 0.08 +  # multiple loans
        np.random.normal(0, 0.03, n_samples)  # noise
    )
    
    # Add drift effect (economic downturn)
    if drift_factor > 0:
        economic_stress = (income < 40000) * drift_factor * 0.2
        default_prob += economic_stress
    
    default_prob = np.clip(default_prob, 0, 0.6)
    default = np.random.binomial(1, default_prob, n_samples)
    
    df = pd.DataFrame({
        # Personal
        'age': age.round().astype(int),
        'annual_income': income.round(2),
        'employment_length_years': employment_length.round(1),
        
        # Financial history
        'existing_loans_count': existing_loans,
        'credit_history_years': credit_history_length.round(1),
        'debt_to_income_ratio': debt_to_income.round(3),
        
        # Credit behavior
        'credit_utilization_ratio': credit_utilization.round(3),
        'payment_history_score': (payment_history_score * 100).round(1),
        'credit_inquiries_6m': inquiries_last_6m,
        
        # Loan details
        'loan_amount': loan_amount.round(2),
        'loan_purpose': loan_purpose,
        'collateral_value': collateral_value.round(2),
        
        # Target
        'default': default
    })
    
    return df

def create_house_price_dataset(n_samples=6000, drift_factor=0.0):
    """Create a realistic house price dataset."""
    np.random.seed(456 + int(drift_factor * 100))
    
    # Property features
    square_feet = np.random.normal(2000 + drift_factor * 200, 800, n_samples).clip(500, 8000)
    bedrooms = np.random.poisson(3, n_samples).clip(1, 8)
    bathrooms = bedrooms * 0.75 + np.random.normal(0, 0.5, n_samples)
    bathrooms = bathrooms.clip(1, 6).round(1)
    
    # Lot and structure
    lot_size = np.random.gamma(2, 0.3, n_samples).clip(0.1, 5.0)  # acres
    year_built = np.random.normal(1990 + drift_factor * 5, 25, n_samples).clip(1900, 2023).astype(int)
    garage_size = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1])
    
    # Location quality
    school_rating = np.random.beta(3, 2, n_samples) * 10  # 0-10 scale
    crime_rate = np.random.exponential(2 + drift_factor, n_samples).clip(0, 20)  # per 1000
    distance_downtown = np.random.gamma(2, 5, n_samples).clip(0.5, 50)  # miles
    
    # Neighborhood amenities
    nearby_parks = np.random.poisson(3, n_samples).clip(0, 15)
    public_transport_score = np.random.beta(2, 3, n_samples) * 10
    shopping_centers = np.random.poisson(2, n_samples).clip(0, 10)
    
    # Economic factors
    property_tax_rate = np.random.normal(1.2 + drift_factor * 0.2, 0.3, n_samples).clip(0.5, 3.0)  # %
    hoa_fees = np.random.exponential(200, n_samples).clip(0, 1000)  # monthly
    
    # House condition
    renovation_year = np.where(
        np.random.random(n_samples) < 0.3,  # 30% renovated
        np.random.randint(year_built, 2024, n_samples),
        year_built
    )
    
    # Calculate price based on features
    base_price = (
        square_feet * 100 +  # $100 per sq ft
        bedrooms * 15000 +  # $15k per bedroom
        bathrooms * 10000 +  # $10k per bathroom
        lot_size * 50000 +  # $50k per acre
        garage_size * 8000 +  # $8k per garage space
        (2024 - year_built) * -500 +  # depreciation
        school_rating * 8000 +  # school premium
        crime_rate * -2000 +  # crime discount
        distance_downtown * -1000 +  # distance discount
        nearby_parks * 2000 +  # parks premium
        public_transport_score * 1000  # transport premium
    )
    
    # Add market effects
    market_multiplier = np.random.normal(1.0 + drift_factor * 0.3, 0.2, n_samples).clip(0.5, 2.0)
    price = base_price * market_multiplier
    
    # Add noise
    price += np.random.normal(0, 20000, n_samples)
    price = price.clip(50000, 2000000)
    
    df = pd.DataFrame({
        # Structure
        'square_feet': square_feet.round().astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'garage_size': garage_size,
        'lot_size_acres': lot_size.round(2),
        
        # Age and condition
        'year_built': year_built,
        'last_renovation_year': renovation_year,
        'age_years': (2024 - year_built),
        
        # Location
        'school_rating': school_rating.round(1),
        'crime_rate_per_1000': crime_rate.round(2),
        'distance_downtown_miles': distance_downtown.round(1),
        
        # Amenities
        'nearby_parks_count': nearby_parks,
        'public_transport_score': public_transport_score.round(1),
        'shopping_centers_count': shopping_centers,
        
        # Costs
        'property_tax_rate': property_tax_rate.round(3),
        'hoa_monthly_fee': hoa_fees.round(2),
        
        # Target
        'price': price.round(2)
    })
    
    return df

def create_ecommerce_conversion_dataset(n_samples=15000, drift_factor=0.0):
    """Create an e-commerce conversion dataset."""
    np.random.seed(789 + int(drift_factor * 100))
    
    # User demographics
    age = np.random.gamma(2, 15, n_samples).clip(16, 80)
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.6, 0.3, 0.1])
    
    # Session behavior
    session_duration = np.random.exponential(5 + drift_factor, n_samples).clip(0.1, 60)  # minutes
    pages_viewed = np.random.poisson(4 + drift_factor, n_samples).clip(1, 50)
    bounce_rate = np.random.beta(2, 3, n_samples)
    
    # Traffic source
    traffic_source = np.random.choice(['organic', 'paid', 'social', 'email', 'direct'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    is_returning_user = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Product interaction
    products_viewed = np.random.poisson(3 + drift_factor, n_samples).clip(1, 20)
    cart_additions = np.random.poisson(1, n_samples).clip(0, 15)
    cart_value = cart_additions * np.random.exponential(50, n_samples)
    
    # Timing
    hour_of_day = np.random.randint(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    is_weekend = (day_of_week >= 5).astype(int)
    
    # Promotions
    discount_offered = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    discount_percentage = np.where(discount_offered, np.random.uniform(5, 50, n_samples), 0)
    
    # Calculate conversion probability
    conversion_prob = (
        0.02 +  # base conversion rate
        (device_type == 'desktop') * 0.03 +  # desktop converts better
        (is_returning_user) * 0.04 +  # returning users convert better
        (traffic_source == 'email') * 0.06 +  # email traffic converts well
        (traffic_source == 'paid') * 0.03 +  # paid traffic converts well
        (session_duration > 10) * 0.05 +  # longer sessions
        (pages_viewed > 5) * 0.03 +  # more engagement
        (cart_additions > 0) * 0.15 +  # items in cart
        (discount_offered) * 0.04 +  # discounts help
        (hour_of_day >= 19) * 0.02 +  # evening shopping
        np.random.normal(0, 0.01, n_samples)  # noise
    )
    
    # Add drift effect (market saturation)
    if drift_factor > 0:
        market_saturation = drift_factor * 0.1
        conversion_prob -= market_saturation
    
    conversion_prob = np.clip(conversion_prob, 0, 0.4)
    converted = np.random.binomial(1, conversion_prob, n_samples)
    
    df = pd.DataFrame({
        # User info
        'user_age': age.round().astype(int),
        'device_type': device_type,
        'is_returning_user': is_returning_user,
        
        # Session
        'session_duration_minutes': session_duration.round(2),
        'pages_viewed': pages_viewed,
        'bounce_rate': bounce_rate.round(3),
        
        # Traffic
        'traffic_source': traffic_source,
        
        # Products
        'products_viewed': products_viewed,
        'cart_additions': cart_additions,
        'cart_value': cart_value.round(2),
        
        # Timing
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        
        # Promotions
        'discount_offered': discount_offered,
        'discount_percentage': discount_percentage.round(1),
        
        # Target
        'converted': converted
    })
    
    return df

def generate_all_datasets():
    """Generate all sample datasets."""
    print("🏭 Generating comprehensive sample datasets...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    datasets = {}
    
    # Customer churn (base and drifted versions)
    print("📱 Creating customer churn datasets...")
    churn_base = create_customer_churn_dataset(10000, drift_factor=0.0)
    churn_drift = create_customer_churn_dataset(2000, drift_factor=0.4)
    
    datasets['customer_churn_train'] = churn_base[:8000]
    datasets['customer_churn_test'] = churn_base[8000:]
    datasets['customer_churn_drift'] = churn_drift
    
    # Credit scoring
    print("💳 Creating credit scoring datasets...")
    credit_base = create_credit_scoring_dataset(8000, drift_factor=0.0)
    credit_drift = create_credit_scoring_dataset(1500, drift_factor=0.3)
    
    datasets['credit_scoring_train'] = credit_base[:6400]
    datasets['credit_scoring_test'] = credit_base[6400:]
    datasets['credit_scoring_drift'] = credit_drift
    
    # House prices
    print("🏠 Creating house price datasets...")
    house_base = create_house_price_dataset(6000, drift_factor=0.0)
    house_drift = create_house_price_dataset(1200, drift_factor=0.5)
    
    datasets['house_prices_train'] = house_base[:4800]
    datasets['house_prices_test'] = house_base[4800:]
    datasets['house_prices_drift'] = house_drift
    
    # E-commerce conversion
    print("🛒 Creating e-commerce conversion datasets...")
    ecomm_base = create_ecommerce_conversion_dataset(15000, drift_factor=0.0)
    ecomm_drift = create_ecommerce_conversion_dataset(3000, drift_factor=0.2)
    
    datasets['ecommerce_conversion_train'] = ecomm_base[:12000]
    datasets['ecommerce_conversion_test'] = ecomm_base[12000:]
    datasets['ecommerce_conversion_drift'] = ecomm_drift
    
    # Save all datasets
    for name, df in datasets.items():
        filepath = data_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {name}: {len(df)} samples → {filepath}")
    
    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'datasets': {
            name: {
                'samples': len(df),
                'features': len(df.columns) - 1,  # excluding target
                'target': get_target_column(name),
                'type': get_dataset_type(name)
            }
            for name, df in datasets.items()
        }
    }
    
    with open(data_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Generated {len(datasets)} datasets with {sum(len(df) for df in datasets.values())} total samples")
    return datasets

def get_target_column(dataset_name):
    """Get the target column for a dataset."""
    if 'churn' in dataset_name:
        return 'churn'
    elif 'credit' in dataset_name:
        return 'default'
    elif 'house' in dataset_name:
        return 'price'
    elif 'ecommerce' in dataset_name:
        return 'converted'
    return 'target'

def get_dataset_type(dataset_name):
    """Get the ML task type for a dataset."""
    if any(x in dataset_name for x in ['churn', 'credit', 'ecommerce']):
        return 'classification'
    elif 'house' in dataset_name:
        return 'regression'
    return 'unknown'

if __name__ == "__main__":
    generate_all_datasets()
