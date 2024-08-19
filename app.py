import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import requests
import plotly.express as px
import numpy as np
import zipfile
import io

nltk.download('stopwords')


# Function to load and preprocess data
def load_and_preprocess_data(file):
    data = pd.read_csv(file)
    if 'CTR' in data.columns:
        data['CTR'] = data['CTR'].str.rstrip('%').astype('float') / 100
    if 'Impressions' in data.columns:
        data['Impressions'] = data['Impressions'].astype(int)
    if 'Clicks' in data.columns:
        data['Clicks'] = data['Clicks'].astype(int)
    if 'Position' in data.columns:
        data['Position'] = data['Position'].astype(float)
    return data


def get_keyword_metrics(keywords, api_key, user_id, country_code):
    url = "https://learnwithhasan.com/wp-json/lwh-user-api/v1/seo/get-bulk-keyword-metrics"
    params = {
        "query": keywords,
        "keywords_count": 20,
        "countryCode": country_code
    }
    headers = {
        "X-Auth-Key": api_key,
        "X-User-ID": user_id
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error(f"API Error: {response.status_code}")
        return None

def fetch_keyword_metrics(keywords, api_key, user_id, country_code):
    all_metrics = []
    for i in range(0, len(keywords), 20):
        batch = keywords[i:i+20]
        keyword_string = ','.join(batch)
        try:
            response = get_keyword_metrics(keyword_string, api_key, user_id, country_code)
            if response and response.get('success'):
                all_metrics.extend(response['result'])
            else:
                st.warning(f"API Warning for batch {i//20 + 1}: {response.get('message', 'Unknown error')}")
                all_metrics.extend([{'keyword': kw, 'searchVolume': 'N/A', 'cpc': 'N/A', 'difficulty': 'N/A'} for kw in batch])
        except Exception as e:
            st.error(f"Error fetching keyword metrics for batch {i//20 + 1}: {str(e)}")
            all_metrics.extend([{'keyword': kw, 'searchVolume': 'N/A', 'cpc': 'N/A', 'difficulty': 'N/A'} for kw in batch])
    
    return all_metrics

def add_keyword_metrics(df, include_metrics, api_key, user_id, country_code):
    if include_metrics and api_key and user_id:
        keywords = df['Top queries'].tolist()
        metrics = fetch_keyword_metrics(keywords, api_key, user_id, country_code)
        metrics_df = pd.DataFrame(metrics)
        df = df.merge(metrics_df, left_on='Top queries', right_on='keyword', how='left')
        df = df.drop('keyword', axis=1)
    return df

# Update all analysis functions to include country_code parameter
def show_top_performing(data, n=20, include_metrics=False, api_key=None, user_id=None, country_code=None):
    st.markdown("---")
    st.subheader(f"Top {n} Performing Queries")
    top_queries = data.sort_values(by='Clicks', ascending=False).head(n)
    top_queries = add_keyword_metrics(top_queries, include_metrics, api_key, user_id, country_code)
    st.dataframe(top_queries)

def show_opportunities(data, min_impressions, max_position, include_metrics=False, api_key=None, user_id=None, country_code=None):
    st.markdown("---")
    st.subheader(f"Keyword Opportunities (Position > {max_position})")
    st.write(f"These are keywords ranking beyond position {max_position} with at least {min_impressions} impressions. "
             "They represent opportunities to improve your content and potentially gain more traffic.")
    opportunities = data[(data['Position'] > max_position) & (data['Impressions'] >= min_impressions)]
    opportunities = opportunities.sort_values(by='Impressions', ascending=False)
    opportunities = add_keyword_metrics(opportunities, include_metrics, api_key, user_id, country_code)
    st.dataframe(opportunities)

def show_quick_wins(data, min_position, max_position, min_impressions, include_metrics=False, api_key=None, user_id=None, country_code=None):
    st.markdown("---")
    st.subheader(f"Quick Wins (Position {min_position}-{max_position})")
    st.write(f"These are keywords ranking between positions {min_position} and {max_position} with at least {min_impressions} impressions. "
             "They are close to the first page or top positions and could be improved with some optimization.")
    quick_wins = data[(data['Position'] >= min_position) & (data['Position'] <= max_position) & (data['Impressions'] >= min_impressions)]
    quick_wins = quick_wins.sort_values(by='Position')
    quick_wins = add_keyword_metrics(quick_wins, include_metrics, api_key, user_id, country_code)
    st.dataframe(quick_wins)

def highlight_low_hanging_fruits(data, include_metrics=False, api_key=None, user_id=None, country_code=None):
    st.markdown("---")
    st.subheader("Low-Hanging Fruits")
    st.write("These queries are in the top 3 positions but have a low CTR, representing quick wins if optimized.")
    low_hanging_fruits = data[(data['Position'] <= 3) & (data['CTR'] < 0.5)]
    low_hanging_fruits = add_keyword_metrics(low_hanging_fruits, include_metrics, api_key, user_id, country_code)
    st.dataframe(low_hanging_fruits)

def identify_question_queries(data, include_metrics=False, api_key=None, user_id=None, country_code=None):
    st.markdown("---")
    st.subheader("Question Queries")
    st.write("These queries are questions that may be targeted with detailed, high-quality content.")
    question_queries = data[data['Top queries'].str.contains('who|what|where|when|why|how', case=False, na=False)]
    question_queries = add_keyword_metrics(question_queries, include_metrics, api_key, user_id, country_code)
    st.dataframe(question_queries)

# Function to generate word cloud
def generate_word_cloud(data):
    st.markdown("---")
    st.subheader("Keyword Word Cloud")
    st.write("This word cloud visualizes the most common words in your top queries. "
             "It can help identify themes and topics that are performing well in search results.")
    text = ' '.join(data['Top queries'])
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    


# New Function: Traffic Potential Estimation
def estimate_traffic_potential(data):
    st.markdown("---")
    st.subheader("Traffic Potential Estimation")
    
    # Calculate the average CTR for top 10 positions
    top_10_data = data[data['Position'] <= 10]
    if len(top_10_data) > 0:
        top_10_ctr = top_10_data['CTR'].mean()
    else:
        top_10_ctr = data['CTR'].mean()  # Fallback if no queries in top 10
    
    # Ensure CTR is always between 0 and 1
    data['CTR'] = data['CTR'].clip(0, 1)
    
    # Calculate potential clicks based on position
    data['Potential CTR'] = np.where(data['Position'] <= 10, data['CTR'], 
                                     np.maximum(data['CTR'], top_10_ctr))
    data['Potential Clicks'] = data['Impressions'] * data['Potential CTR']
    data['Click Potential Increase'] = np.maximum(data['Potential Clicks'] - data['Clicks'], 0)
    
    # Filter for meaningful improvements
    potential_data = data[data['Click Potential Increase'] > 1].sort_values(by='Click Potential Increase', ascending=False)
    
    # Display results
    st.write(f"Average CTR for top 10 positions: {top_10_ctr:.2%}")
    st.dataframe(potential_data[['Top queries', 'Position', 'Clicks', 'Potential Clicks', 'Click Potential Increase']].head(20))
    
    # Summary statistics
    total_current_clicks = data['Clicks'].sum()
    total_potential_clicks = data['Potential Clicks'].sum()
    total_increase = total_potential_clicks - total_current_clicks
    
    st.write(f"Total current clicks: {total_current_clicks:,.0f}")
    st.write(f"Total potential clicks: {total_potential_clicks:,.0f}")
    st.write(f"Potential total increase: {total_increase:,.0f} ({(total_increase/total_current_clicks)*100:.2f}% increase)")
    
    st.markdown("""
    #### Understanding the Traffic Potential Estimation

    This estimation provides a rough idea of how much your click traffic might increase if lower-ranking queries performed better. Here's how we calculate it:

    1. **Average CTR Calculation**: We first calculate the average Click-Through Rate (CTR) for queries ranking in the top 10 positions. CTR is the percentage of impressions that resulted in a click.

    2. **Potential CTR Assignment**: For queries ranking below position 10, we assume they could potentially achieve this average top 10 CTR. For queries already in the top 10, we keep their current CTR.

    3. **Potential Clicks Calculation**: We multiply the number of impressions for each query by its potential CTR to get the potential number of clicks.

    4. **Click Increase Estimation**: We subtract the current clicks from the potential clicks to estimate the potential increase.

    5. **Filtering**: We only show queries where the potential increase is more than 1 click, to focus on meaningful improvements.

    #### Important Notes:

    - This is a very rough estimate and not an accurate prediction. Actual results can vary significantly.
    - Many factors affect CTR, including the query itself, competition, and SERP features, which this simple model doesn't account for.
    - Improving rankings is challenging and takes time and effort.
    - This estimate assumes you could achieve the average CTR of top-ranking queries for all your lower-ranking queries, which is often unrealistic.
    - Use this as a general guide for potential opportunities, not as a guarantee of traffic increases.

    Always combine this data with other SEO metrics and your expert knowledge of your content and audience for a more comprehensive understanding of your optimization opportunities.
    """)



# Updated Country Performance Dashboard function
def country_performance_dashboard(country_data):
    st.markdown("---")
    st.subheader("Country Performance Dashboard")
    st.dataframe(country_data.sort_values(by='Clicks', ascending=False))
    
    # Comprehensive dictionary to map country names to ISO codes
    country_code_map = {
        'India': 'IND', 'United States': 'USA', 'Pakistan': 'PAK', 'Bangladesh': 'BGD',
        'Nigeria': 'NGA', 'United Kingdom': 'GBR', 'Morocco': 'MAR', 'Germany': 'DEU',
        'Indonesia': 'IDN', 'Canada': 'CAN', 'Algeria': 'DZA', 'Egypt': 'EGY',
        'Australia': 'AUS', 'France': 'FRA', 'Vietnam': 'VNM', 'Brazil': 'BRA',
        'Sri Lanka': 'LKA', 'Spain': 'ESP', 'United Arab Emirates': 'ARE', 'Netherlands': 'NLD',
        'Italy': 'ITA', 'Turkey': 'TUR', 'Saudi Arabia': 'SAU', 'Kenya': 'KEN',
        'Poland': 'POL', 'Philippines': 'PHL', 'Malaysia': 'MYS', 'South Korea': 'KOR',
        'Japan': 'JPN', 'South Africa': 'ZAF', 'Thailand': 'THA', 'Singapore': 'SGP',
        'Iran': 'IRN', 'Israel': 'ISR', 'Mexico': 'MEX', 'Russia': 'RUS',
        'Sweden': 'SWE', 'Taiwan': 'TWN', 'Ghana': 'GHA', 'Tunisia': 'TUN',
        'Romania': 'ROU', 'Nepal': 'NPL', 'Ukraine': 'UKR', 'Belgium': 'BEL',
        'Portugal': 'PRT', 'China': 'CHN', 'Hong Kong': 'HKG', 'Colombia': 'COL',
        'Serbia': 'SRB', 'Denmark': 'DNK', 'Switzerland': 'CHE', 'Lebanon': 'LBN',
        'Jordan': 'JOR', 'Austria': 'AUT', 'Ethiopia': 'ETH', 'Hungary': 'HUN',
        'Czechia': 'CZE', 'Ireland': 'IRL', 'Argentina': 'ARG', 'Norway': 'NOR',
        'Peru': 'PER', 'Greece': 'GRC', 'Uganda': 'UGA', 'Bulgaria': 'BGR',
        'New Zealand': 'NZL', 'Iraq': 'IRQ', 'Finland': 'FIN', 'Qatar': 'QAT',
        'Somalia': 'SOM', 'Cameroon': 'CMR', 'Tanzania': 'TZA', 'Chile': 'CHL',
        'Kuwait': 'KWT', 'Yemen': 'YEM', 'Guatemala': 'GTM', 'Venezuela': 'VEN',
        'Slovakia': 'SVK', 'Cambodia': 'KHM', 'Cyprus': 'CYP', 'Kazakhstan': 'KAZ',
        'Oman': 'OMN', 'Bosnia & Herzegovina': 'BIH', 'Bahrain': 'BHR', 'Dominican Republic': 'DOM',
        'Latvia': 'LVA', 'Estonia': 'EST', 'Armenia': 'ARM', "Côte d'Ivoire": 'CIV',
        'Ecuador': 'ECU', 'Albania': 'ALB', 'Togo': 'TGO', 'Palestine': 'PSE',
        'Lithuania': 'LTU', 'Belarus': 'BLR', 'Croatia': 'HRV', 'Slovenia': 'SVN',
        'Rwanda': 'RWA', 'Benin': 'BEN', 'Costa Rica': 'CRI', 'Macedonia': 'MKD',
        'Luxembourg': 'LUX', 'Georgia': 'GEO', 'Bolivia': 'BOL', 'Azerbaijan': 'AZE',
        'Libya': 'LBY', 'Panama': 'PAN', 'Syria': 'SYR', 'Zambia': 'ZMB',
        'Zimbabwe': 'ZWE', 'Trinidad & Tobago': 'TTO', 'Barbados': 'BRB', 'Uzbekistan': 'UZB',
        'Uruguay': 'URY', 'Moldova': 'MDA', 'Mauritius': 'MUS', 'Sudan': 'SDN',
        'Malta': 'MLT', 'Madagascar': 'MDG', 'Congo - Kinshasa': 'COD', 'Puerto Rico': 'PRI',
        'Senegal': 'SEN', 'Myanmar (Burma)': 'MMR', 'Maldives': 'MDV', 'Burkina Faso': 'BFA',
        'Gambia': 'GMB', 'South Sudan': 'SSD', 'Jamaica': 'JAM', 'Angola': 'AGO',
        'Mozambique': 'MOZ', 'Malawi': 'MWI', 'Laos': 'LAO', 'Iceland': 'ISL',
        'Grenada': 'GRD', 'Botswana': 'BWA', 'Afghanistan': 'AFG', 'Congo - Brazzaville': 'COG',
        'Macau': 'MAC', 'Bhutan': 'BTN', 'Paraguay': 'PRY', 'Mongolia': 'MNG',
        'Nicaragua': 'NIC', 'Kyrgyzstan': 'KGZ', 'St. Lucia': 'LCA', 'Turkmenistan': 'TKM',
        'Papua New Guinea': 'PNG', 'Swaziland': 'SWZ', 'Burundi': 'BDI', 'Liberia': 'LBR',
        'El Salvador': 'SLV', 'Guyana': 'GUY', 'Belize': 'BLZ', 'Montenegro': 'MNE',
        'Réunion': 'REU', 'Cayman Islands': 'CYM', 'Suriname': 'SUR', 'Namibia': 'NAM',
        'St. Vincent & Grenadines': 'VCT', 'Cuba': 'CUB', 'Curaçao': 'CUW', 'Tajikistan': 'TJK',
        'Haiti': 'HTI', 'Andorra': 'AND', 'Martinique': 'MTQ', 'Mali': 'MLI',
        'Mauritania': 'MRT', 'French Polynesia': 'PYF', 'Guernsey': 'GGY', 'Djibouti': 'DJI',
        'French Guiana': 'GUF', 'Chad': 'TCD', 'Faroe Islands': 'FRO', 'Guinea': 'GIN',
        'Liechtenstein': 'LIE', 'Vanuatu': 'VUT', 'Niger': 'NER', 'Timor-Leste': 'TLS',
        'Honduras': 'HND', 'Bahamas': 'BHS', 'Seychelles': 'SYC', 'Brunei': 'BRN',
        'Gabon': 'GAB', 'St. Kitts & Nevis': 'KNA', 'Bermuda': 'BMU', 'Antigua & Barbuda': 'ATG',
        'Kosovo': 'XKX', 'Fiji': 'FJI', 'Guadeloupe': 'GLP', 'Aruba': 'ABW',
        'British Virgin Islands': 'VGB', 'Guam': 'GUM', 'Sint Maarten': 'SXM', 'Isle of Man': 'IMN',
        'Jersey': 'JEY', 'Turks & Caicos Islands': 'TCA', 'Dominica': 'DMA', 'Cape Verde': 'CPV',
        'New Caledonia': 'NCL', 'Gibraltar': 'GIB', 'Lesotho': 'LSO', 'Sierra Leone': 'SLE',
        'Anguilla': 'AIA', 'Caribbean Netherlands': 'BES', 'Mayotte': 'MYT', 'Monaco': 'MCO',
        'San Marino': 'SMR', 'U.S. Virgin Islands': 'VIR', 'Micronesia': 'FSM',
        'São Tomé & Príncipe': 'STP', 'Equatorial Guinea': 'GNQ', 'Samoa': 'WSM',
        'Montserrat': 'MSR', 'Western Sahara': 'ESH', 'Antarctica': 'ATA', 'North Korea': 'PRK',
        'Greenland': 'GRL', 'Northern Mariana Islands': 'MNP', 'Marshall Islands': 'MHL',
        'Central African Republic': 'CAF', 'Tonga': 'TON', 'Solomon Islands': 'SLB',
        'Palau': 'PLW', 'Kiribati': 'KIR', 'Eritrea': 'ERI', 'St. Pierre & Miquelon': 'SPM',
        'St. Martin': 'MAF', 'Comoros': 'COM', 'St. Helena': 'SHN', 'American Samoa': 'ASM',
        'Guinea-Bissau': 'GNB', 'Svalbard & Jan Mayen': 'SJM', 'St. Barthélemy': 'BLM',
        'Åland Islands': 'ALA', 'Wallis & Futuna': 'WLF', 'Tuvalu': 'TUV',
        'Unknown Region': 'UNK'
    }
    
    # Function to get ISO code or return original name if not found
    def get_iso_code(country):
        return country_code_map.get(country, country)
    
    # Apply the mapping to the Country column
    country_data['Country_Code'] = country_data['Country'].apply(get_iso_code)
    
    # Function to create choropleth map
    def create_choropleth(data, metric, title):
        fig = px.choropleth(data, 
                            locations="Country_Code", 
                            locationmode="ISO-3",
                            color=metric, 
                            hover_name="Country", 
                            color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(title=title)
        return fig
    
    # Create and display Clicks map
    clicks_fig = create_choropleth(country_data, "Clicks", "Clicks by Country")
    st.plotly_chart(clicks_fig)
    
    # Create and display Impressions map
    impressions_fig = create_choropleth(country_data, "Impressions", "Impressions by Country")
    st.plotly_chart(impressions_fig)
    
    # Display countries that might not be showing on the map
    unknown_countries = country_data[country_data['Country'] == country_data['Country_Code']]
    if not unknown_countries.empty:
        st.write("Countries that might not be showing on the map:")
        st.write(unknown_countries['Country'].tolist())

# New Function: Top Opportunities by Country
def top_opportunities_by_country(country_data):
    st.markdown("---")

    st.subheader("Top Opportunities by Country")
    st.write("These countries have high impressions but low CTR, representing opportunities to improve content for these audiences.")

    low_ctr_countries = country_data[country_data['CTR'] < country_data['CTR'].median()]
    st.dataframe(low_ctr_countries.sort_values(by='Impressions', ascending=False))

# New Function: Top Pages Analysis
def top_pages_analysis(page_data):
    st.markdown("---")

    st.subheader("Top Pages Analysis")
    st.write("These are the top pages driving traffic, sorted by the number of clicks.")

    st.dataframe(page_data.sort_values(by='Clicks', ascending=False))

# New Function: Pages Needing Optimization
def pages_needing_optimization(page_data):
    st.markdown("---")

    st.subheader("Pages Needing Optimization")
    st.write("These pages have high impressions but low CTR, representing opportunities for optimization.")

    low_ctr_pages = page_data[page_data['CTR'] < 0.5]
    st.dataframe(low_ctr_pages.sort_values(by='Impressions', ascending=False))


# Main Streamlit app
def main():
    st.title("Advanced SEO Data Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose your zip file containing Query, Country, and Page data", type="zip")
    
    # Sidebar for user inputs
    st.sidebar.header("Analysis Parameters")
    top_n = st.sidebar.slider("Number of top queries to show", 5, 50, 20)
    min_impressions = st.sidebar.slider("Minimum Impressions (Global)", 100, 10000, 1000)
    max_position_opp = st.sidebar.slider("Maximum Position for Keyword Opportunities", 50, 100, 60)
    min_position_quick = st.sidebar.slider("Minimum Position for Quick Wins", 5, 20, 11)
    max_position_quick = st.sidebar.slider("Maximum Position for Quick Wins", 20, 50, 20)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⭐ For Power Members")

    # Checkbox for keyword metrics
    include_metrics = st.sidebar.checkbox("Include additional keyword metrics")
    
    api_key = None
    user_id = None
    country_code = None
    if include_metrics:
        api_key = st.sidebar.text_input("Enter your API Key", type="password")
        user_id = st.sidebar.text_input("Enter your User ID")
        country_options = {
            "United States": "US","United Kingdom": "UK","India": "IN", "Argentina": "AR", "Australia": "AU", "Brazil": "BR", "Canada": "CA", 
            "Germany": "DE", "Spain": "ES", "France": "FR", "Ireland": "IE", 
             "Italy": "IT", "Mexico": "MX", "Netherlands": "NL", 
            "New Zealand": "NZ", "Singapore": "SG", "Ukraine": "UA", 
             "South Africa": "ZA"
        }
        country_name = st.sidebar.selectbox("Select Country", list(country_options.keys()))
        country_code = country_options[country_name]
    
    # Button to start analysis
    start_analysis = st.button("Start Analysis")
    
    if start_analysis and uploaded_file is not None:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            
            gsc_file = next((f for f in file_names if f.startswith('Quer')), None)
            country_file = next((f for f in file_names if f.startswith('Countr')), None)
            page_file = next((f for f in file_names if f.startswith('Page')), None)
            
            if gsc_file and country_file and page_file:
                gsc_data = load_and_preprocess_data(io.BytesIO(zip_ref.read(gsc_file)))
                country_data = load_and_preprocess_data(io.BytesIO(zip_ref.read(country_file)))
                page_data = load_and_preprocess_data(io.BytesIO(zip_ref.read(page_file)))
                
                st.write("Data successfully loaded and processed.")
                
                # Main content
                show_top_performing(gsc_data, top_n, include_metrics, api_key, user_id, country_code)
                show_opportunities(gsc_data, min_impressions, max_position_opp, include_metrics, api_key, user_id, country_code)
                show_quick_wins(gsc_data, min_position_quick, max_position_quick, min_impressions, include_metrics, api_key, user_id, country_code)
                generate_word_cloud(gsc_data)
                
                # Additional features
                highlight_low_hanging_fruits(gsc_data, include_metrics, api_key, user_id, country_code)
                identify_question_queries(gsc_data, include_metrics, api_key, user_id, country_code)
                estimate_traffic_potential(gsc_data)
                
                # Country-specific analysis
                country_performance_dashboard(country_data)
                top_opportunities_by_country(country_data)
                
                # Page-specific analysis
                top_pages_analysis(page_data)
                pages_needing_optimization(page_data)
                
            else:
                st.error("Please make sure your zip file contains files named gsc_*.csv, country_*.csv, and page_*.csv")
    elif start_analysis:
        st.error("Please upload a zip file containing your Google Search Console, Country, and Page data CSV files.")

if __name__ == "__main__":
    main()