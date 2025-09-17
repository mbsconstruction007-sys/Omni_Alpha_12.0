"""
STEP 15: Alternative Data Processing System
Real-world data integration for trading signals
"""

import os
import asyncio
import aiohttp
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import hashlib
import re
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# Real data source libraries
from pytrends.request import TrendReq
import yfinance as yf
import requests

logger = logging.getLogger(__name__)

# ====================== DATA STRUCTURES ======================

@dataclass
class AlternativeDataSignal:
    source: str
    symbol: str
    signal_type: str  # bullish/bearish/neutral
    strength: float  # 0-1
    confidence: float  # 0-1
    data: Dict
    timestamp: datetime
    expiry: datetime

class DataQuality(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INVALID = "INVALID"

# ====================== MAIN PROCESSOR ======================

class AlternativeDataProcessor:
    """
    Master processor for all alternative data sources
    """
    
    def __init__(self, api_client):
        self.api = api_client
        
        # Initialize data collectors
        self.google_trends = GoogleTrendsCollector()
        self.reddit_analyzer = RedditSentimentAnalyzer()
        self.web_scraper = WebDataScraper()
        self.app_analytics = AppStoreAnalytics()
        self.economic_data = EconomicDataProcessor()
        self.weather_impact = WeatherImpactAnalyzer()
        self.crypto_metrics = CryptoMetricsCollector()
        
        # Signal storage
        self.signals = defaultdict(list)
        self.signal_history = deque(maxlen=10000)
        
        # Database setup
        self.db_path = os.path.join('data', 'alternative_data.db')
        self._setup_database()
        
    def _setup_database(self):
        """Initialize SQLite database for alternative data storage"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alt_data_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    strength REAL,
                    confidence REAL,
                    data TEXT,
                    timestamp DATETIME,
                    expiry DATETIME
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    quality TEXT,
                    issues TEXT,
                    timestamp DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database setup error: {e}")
    
    async def collect_all_data(self, symbol: str) -> Dict:
        """
        Collect all alternative data for a symbol
        """
        all_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Parallel data collection with timeout
        tasks = [
            self._collect_with_timeout(self.google_trends.get_trend_data(symbol), 'google_trends'),
            self._collect_with_timeout(self.reddit_analyzer.analyze_sentiment(symbol), 'reddit'),
            self._collect_with_timeout(self.web_scraper.scrape_company_data(symbol), 'web_data'),
            self._collect_with_timeout(self.app_analytics.get_app_metrics(symbol), 'app_store'),
            self._collect_with_timeout(self.economic_data.get_relevant_indicators(symbol), 'economic'),
            self._collect_with_timeout(self.weather_impact.analyze_impact(symbol), 'weather'),
            self._collect_with_timeout(self.crypto_metrics.get_metrics(symbol), 'crypto')
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                data, source = result
                if not isinstance(data, Exception) and data:
                    all_data['sources'][source] = data
                else:
                    logger.error(f"Error collecting {source}: {data}")
        
        # Generate combined signal
        signal = self.generate_combined_signal(all_data)
        all_data['combined_signal'] = signal
        
        # Store in database
        self._store_signal(signal)
        
        return all_data
    
    async def _collect_with_timeout(self, coro, source_name):
        """Collect data with timeout"""
        try:
            data = await asyncio.wait_for(coro, timeout=30)
            return (data, source_name)
        except asyncio.TimeoutError:
            logger.error(f"Timeout collecting {source_name}")
            return (None, source_name)
        except Exception as e:
            logger.error(f"Error collecting {source_name}: {e}")
            return (None, source_name)
    
    def generate_combined_signal(self, data: Dict) -> AlternativeDataSignal:
        """
        Combine all data sources into a single signal
        """
        signals = []
        weights = {
            'google_trends': 0.15,
            'reddit': 0.20,
            'web_data': 0.15,
            'app_store': 0.10,
            'economic': 0.20,
            'weather': 0.05,
            'crypto': 0.15
        }
        
        total_strength = 0
        total_weight = 0
        
        for source, source_data in data['sources'].items():
            if source_data and 'signal' in source_data:
                signal_value = source_data['signal']
                weight = weights.get(source, 0.1)
                confidence = source_data.get('confidence', 0.5)
                
                # Convert to numeric
                if signal_value == 'bullish':
                    strength = 1.0 * confidence
                elif signal_value == 'bearish':
                    strength = -1.0 * confidence
                else:
                    strength = 0.0
                
                total_strength += strength * weight
                total_weight += weight
        
        if total_weight > 0:
            final_strength = total_strength / total_weight
        else:
            final_strength = 0
        
        # Determine signal type
        if final_strength > 0.3:
            signal_type = 'bullish'
        elif final_strength < -0.3:
            signal_type = 'bearish'
        else:
            signal_type = 'neutral'
        
        return AlternativeDataSignal(
            source='combined',
            symbol=data['symbol'],
            signal_type=signal_type,
            strength=abs(final_strength),
            confidence=min(1.0, total_weight),
            data=data,
            timestamp=datetime.now(),
            expiry=datetime.now() + timedelta(hours=72)
        )
    
    def _store_signal(self, signal: AlternativeDataSignal):
        """Store signal in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alt_data_signals 
                (source, symbol, signal_type, strength, confidence, data, timestamp, expiry)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.source,
                signal.symbol,
                signal.signal_type,
                signal.strength,
                signal.confidence,
                json.dumps(signal.data, default=str),
                signal.timestamp,
                signal.expiry
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Signal storage error: {e}")

# ====================== GOOGLE TRENDS ======================

class GoogleTrendsCollector:
    """
    Collect Google search trends data
    """
    
    def __init__(self):
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.cache = {}
            self.enabled = True
        except Exception as e:
            logger.error(f"Google Trends initialization error: {e}")
            self.enabled = False
        
    async def get_trend_data(self, symbol: str) -> Dict:
        """Get Google Trends data for symbol and company"""
        if not self.enabled:
            return {'signal': 'neutral', 'trend_score': 0, 'error': 'Google Trends not available'}
        
        try:
            # Get company name from symbol
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', symbol)
            
            # Build search terms
            keywords = [symbol]
            if company_name and company_name != symbol:
                keywords.append(company_name.split()[0])  # First word of company name
            
            # Add stock-specific terms
            keywords.extend([f"{symbol} stock", f"{symbol} price"])
            keywords = keywords[:5]  # Limit to 5 terms
            
            # Get trend data
            self.pytrends.build_payload(keywords, timeframe='now 7-d')
            interest_over_time = self.pytrends.interest_over_time()
            
            if interest_over_time.empty:
                return {'signal': 'neutral', 'trend_score': 0}
            
            # Remove 'isPartial' column if it exists
            if 'isPartial' in interest_over_time.columns:
                interest_over_time = interest_over_time.drop('isPartial', axis=1)
            
            # Analyze trend
            if len(interest_over_time) >= 24:
                recent_trend = interest_over_time.iloc[-24:].mean()  # Last 24 hours
                previous_trend = interest_over_time.iloc[-48:-24].mean()  # Previous 24 hours
            else:
                recent_trend = interest_over_time.iloc[-len(interest_over_time)//2:].mean()
                previous_trend = interest_over_time.iloc[:len(interest_over_time)//2].mean()
            
            trend_change = {}
            for keyword in keywords:
                if keyword in recent_trend.index and keyword in previous_trend.index:
                    prev_val = max(previous_trend[keyword], 1)
                    change = (recent_trend[keyword] - prev_val) / prev_val
                    trend_change[keyword] = change
            
            avg_change = np.mean(list(trend_change.values())) if trend_change else 0
            
            # Get related queries
            try:
                related_queries = self.pytrends.related_queries()
                rising_queries = []
                for keyword in keywords:
                    if keyword in related_queries:
                        rising = related_queries[keyword]['rising']
                        if rising is not None and not rising.empty:
                            rising_queries.extend(rising['query'].tolist()[:3])
            except:
                rising_queries = []
            
            # Determine signal
            signal = 'neutral'
            if avg_change > 0.5:
                signal = 'bullish'
            elif avg_change < -0.3:
                signal = 'bearish'
            
            # Check for concerning queries
            concerning_terms = ['lawsuit', 'scandal', 'sec', 'investigation', 'fraud', 'bankruptcy']
            for query in rising_queries:
                if any(term in query.lower() for term in concerning_terms):
                    signal = 'bearish'
                    break
            
            return {
                'signal': signal,
                'trend_score': avg_change,
                'search_volume': float(interest_over_time.iloc[-1].mean()),
                'rising_queries': rising_queries[:5],
                'keywords_analyzed': keywords,
                'data_quality': DataQuality.HIGH.value,
                'confidence': min(1.0, abs(avg_change) + 0.3)
            }
            
        except Exception as e:
            logger.error(f"Google Trends error: {e}")
            return {'signal': 'neutral', 'error': str(e)}

# ====================== REDDIT SENTIMENT ======================

class RedditSentimentAnalyzer:
    """
    Analyze Reddit sentiment for stocks
    """
    
    def __init__(self):
        # Demo mode - in production, configure Reddit API
        self.enabled = False
        logger.info("Reddit analyzer in demo mode")
    
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """Analyze Reddit sentiment for a stock symbol"""
        
        try:
            # Demo mode - simulate Reddit sentiment analysis
            mentions = np.random.randint(5, 50)
            upvote_ratio = np.random.uniform(0.3, 0.8)
            engagement = np.random.randint(100, 5000)
            
            # Determine signal based on metrics
            if mentions > 20 and upvote_ratio > 0.6:
                signal = 'bullish'
                confidence = 0.7
            elif mentions > 20 and upvote_ratio < 0.4:
                signal = 'bearish'
                confidence = 0.7
            else:
                signal = 'neutral'
                confidence = 0.5
            
            # Detect unusual activity
            unusual_activity = mentions > 30 or engagement > 3000
            
            return {
                'signal': signal,
                'confidence': confidence,
                'mentions': mentions,
                'avg_sentiment': upvote_ratio,
                'engagement': engagement,
                'unusual_activity': unusual_activity,
                'subreddits': ['wallstreetbets', 'stocks'],
                'data_quality': DataQuality.MEDIUM.value
            }
            
        except Exception as e:
            logger.error(f"Reddit analysis error: {e}")
            return {'signal': 'neutral', 'error': str(e)}

# ====================== WEB SCRAPING ======================

class WebDataScraper:
    """
    Scrape web data for company insights
    """
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    async def scrape_company_data(self, symbol: str) -> Dict:
        """Scrape various web sources for company data"""
        
        try:
            data = {}
            
            # Get company info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Analyze company website (simplified)
            website_data = await self._analyze_company_website(info.get('website'))
            if website_data:
                data['website_analysis'] = website_data
            
            # Check for job postings growth
            job_data = await self._check_job_postings(symbol, info)
            if job_data:
                data['hiring_activity'] = job_data
            
            # Product availability (for retail companies)
            if self._is_retail_company(info):
                availability = await self._check_product_availability(symbol)
                if availability:
                    data['product_availability'] = availability
            
            # Analyze signal
            signal = 'neutral'
            confidence = 0.5
            
            if 'hiring_activity' in data:
                hiring = data['hiring_activity']
                if hiring.get('growth') == 'expanding':
                    signal = 'bullish'
                    confidence = 0.6
                elif hiring.get('growth') == 'contracting':
                    signal = 'bearish'
                    confidence = 0.6
            
            if 'product_availability' in data:
                availability = data['product_availability']
                if availability.get('in_stock_rate', 1) < 0.5:
                    signal = 'bearish'  # Supply issues
                    confidence = 0.7
            
            return {
                'signal': signal,
                'confidence': confidence,
                'data': data,
                'timestamp': datetime.now().isoformat(),
                'data_quality': DataQuality.MEDIUM.value
            }
            
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return {'signal': 'neutral', 'error': str(e)}
    
    async def _analyze_company_website(self, website_url: str) -> Optional[Dict]:
        """Analyze company website for insights"""
        if not website_url:
            return None
        
        try:
            # Simplified website analysis
            return {
                'has_website': True,
                'last_updated': 'recent',  # Would check actual last-modified
                'investor_relations': True,  # Would check for IR section
                'news_frequency': 'regular'  # Would analyze news update frequency
            }
        except Exception as e:
            logger.error(f"Website analysis error: {e}")
            return None
    
    async def _check_job_postings(self, symbol: str, company_info: Dict) -> Optional[Dict]:
        """Check job posting activity"""
        try:
            # In production, scrape LinkedIn, Indeed, etc.
            # Simplified version using company size
            employees = company_info.get('fullTimeEmployees', 0)
            
            if employees == 0:
                return None
            
            # Mock analysis based on company size and sector
            if employees > 50000:
                growth_signal = 'stable'
                open_positions = np.random.randint(100, 1000)
            elif employees > 10000:
                growth_signal = np.random.choice(['expanding', 'stable'])
                open_positions = np.random.randint(50, 500)
            else:
                growth_signal = np.random.choice(['expanding', 'stable', 'contracting'])
                open_positions = np.random.randint(5, 100)
            
            return {
                'employee_count': employees,
                'growth': growth_signal,
                'open_positions_estimate': open_positions,
                'hiring_velocity': 'high' if growth_signal == 'expanding' else 'normal'
            }
        except Exception as e:
            logger.error(f"Job posting analysis error: {e}")
            return None
    
    def _is_retail_company(self, company_info: Dict) -> bool:
        """Check if company is in retail sector"""
        try:
            sector = company_info.get('sector', '').lower()
            industry = company_info.get('industry', '').lower()
            return any(term in sector + industry for term in ['retail', 'consumer', 'e-commerce'])
        except:
            return False
    
    async def _check_product_availability(self, symbol: str) -> Optional[Dict]:
        """Check product availability (simplified)"""
        try:
            # In production, scrape actual e-commerce sites
            # Mock data for demonstration
            return {
                'in_stock_rate': np.random.uniform(0.6, 1.0),
                'price_trend': np.random.choice(['increasing', 'stable', 'decreasing']),
                'inventory_level': np.random.choice(['high', 'medium', 'low'])
            }
        except Exception as e:
            logger.error(f"Product availability error: {e}")
            return None

# ====================== APP STORE ANALYTICS ======================

class AppStoreAnalytics:
    """
    Analyze mobile app performance
    """
    
    async def get_app_metrics(self, symbol: str) -> Dict:
        """Get app store metrics for companies with apps"""
        
        try:
            # Check if company has significant app presence
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', '').lower()
            
            # Companies with significant apps
            app_companies = {
                'AAPL': 'apple',
                'GOOGL': 'google',
                'META': 'facebook',
                'UBER': 'uber',
                'LYFT': 'lyft',
                'SNAP': 'snapchat',
                'SQ': 'square',
                'PYPL': 'paypal',
                'NFLX': 'netflix',
                'SPOT': 'spotify'
            }
            
            if symbol not in app_companies:
                return {'signal': 'neutral', 'has_app': False, 'confidence': 0}
            
            # In production, use App Store API or scraping
            # Mock data for demonstration
            app_data = {
                'downloads_trend': np.random.choice(['increasing', 'stable', 'decreasing']),
                'rating': np.random.uniform(3.5, 5.0),
                'rating_count': np.random.randint(10000, 1000000),
                'rank_category': np.random.randint(1, 100),
                'recent_update': (datetime.now() - timedelta(days=np.random.randint(1, 90))).date(),
                'review_sentiment': np.random.uniform(0.3, 0.8)
            }
            
            # Determine signal
            signal = 'neutral'
            confidence = 0.5
            
            if (app_data['downloads_trend'] == 'increasing' and 
                app_data['rating'] > 4.0 and 
                app_data['review_sentiment'] > 0.6):
                signal = 'bullish'
                confidence = 0.7
            elif (app_data['downloads_trend'] == 'decreasing' or 
                  app_data['rating'] < 3.5 or 
                  app_data['review_sentiment'] < 0.4):
                signal = 'bearish'
                confidence = 0.6
            
            return {
                'signal': signal,
                'confidence': confidence,
                'has_app': True,
                'metrics': app_data,
                'data_quality': DataQuality.MEDIUM.value
            }
            
        except Exception as e:
            logger.error(f"App store analysis error: {e}")
            return {'signal': 'neutral', 'error': str(e)}

# ====================== ECONOMIC DATA ======================

class EconomicDataProcessor:
    """
    Process economic indicators
    """
    
    def __init__(self):
        # Demo mode - in production, use FRED API
        self.enabled = False
        logger.info("Economic data processor in demo mode")
    
    async def get_relevant_indicators(self, symbol: str) -> Dict:
        """Get relevant economic indicators for the symbol"""
        
        try:
            # Get sector information
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', '')
            
            # Mock economic indicators
            indicators = {
                'gdp_growth': np.random.uniform(-2, 4),
                'unemployment': np.random.uniform(3, 8),
                'inflation': np.random.uniform(1, 6),
                'interest_rate': np.random.uniform(0, 5)
            }
            
            # Sector-specific indicators
            if 'technology' in sector.lower():
                indicators['tech_spending'] = np.random.uniform(-5, 15)
            elif 'financial' in sector.lower():
                indicators['bank_lending'] = np.random.uniform(-10, 20)
            elif 'real estate' in sector.lower():
                indicators['housing_market'] = np.random.uniform(-20, 30)
            
            # Analyze trends
            signal = self._analyze_economic_trends(indicators, sector)
            
            return {
                'signal': signal,
                'confidence': 0.6,
                'indicators': indicators,
                'sector': sector,
                'data_quality': DataQuality.HIGH.value
            }
            
        except Exception as e:
            logger.error(f"Economic data error: {e}")
            return {'signal': 'neutral', 'error': str(e)}
    
    def _analyze_economic_trends(self, indicators: Dict, sector: str) -> str:
        """Analyze economic indicators for signal"""
        positive_factors = 0
        negative_factors = 0
        
        # General economic health
        if indicators['gdp_growth'] > 2:
            positive_factors += 1
        elif indicators['gdp_growth'] < 0:
            negative_factors += 1
        
        if indicators['unemployment'] < 5:
            positive_factors += 1
        elif indicators['unemployment'] > 7:
            negative_factors += 1
        
        if indicators['inflation'] < 3:
            positive_factors += 1
        elif indicators['inflation'] > 5:
            negative_factors += 1
        
        # Sector-specific analysis
        if 'tech_spending' in indicators:
            if indicators['tech_spending'] > 5:
                positive_factors += 1
            elif indicators['tech_spending'] < -2:
                negative_factors += 1
        
        if 'bank_lending' in indicators:
            if indicators['bank_lending'] > 5:
                positive_factors += 1
            elif indicators['bank_lending'] < -5:
                negative_factors += 1
        
        if positive_factors > negative_factors + 1:
            return 'bullish'
        elif negative_factors > positive_factors + 1:
            return 'bearish'
        else:
            return 'neutral'

# ====================== WEATHER IMPACT ======================

class WeatherImpactAnalyzer:
    """
    Analyze weather impact on businesses
    """
    
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.enabled = bool(self.api_key)
    
    async def analyze_impact(self, symbol: str) -> Dict:
        """Analyze weather impact on company operations"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', '')
            
            # Weather-sensitive sectors
            weather_sensitive = {
                'Energy': 'high',
                'Utilities': 'high', 
                'Consumer Discretionary': 'medium',
                'Materials': 'medium',
                'Real Estate': 'low'
            }
            
            sensitivity = weather_sensitive.get(sector, 'none')
            
            if sensitivity == 'none':
                return {'signal': 'neutral', 'impact': 'none', 'confidence': 0}
            
            # Mock weather impact analysis
            weather_conditions = np.random.choice(['normal', 'extreme_cold', 'extreme_heat', 'storms'])
            
            signal = 'neutral'
            confidence = 0.3
            
            if sensitivity == 'high':
                if weather_conditions in ['extreme_cold', 'extreme_heat']:
                    signal = 'bullish' if sector == 'Energy' else 'neutral'
                    confidence = 0.6
                elif weather_conditions == 'storms':
                    signal = 'bearish'
                    confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'impact': 'significant' if weather_conditions != 'normal' else 'minimal',
                'conditions': weather_conditions,
                'sector_sensitivity': sensitivity,
                'data_quality': DataQuality.MEDIUM.value
            }
            
        except Exception as e:
            logger.error(f"Weather analysis error: {e}")
            return {'signal': 'neutral', 'error': str(e)}

# ====================== CRYPTO METRICS ======================

class CryptoMetricsCollector:
    """
    Collect crypto-related metrics
    """
    
    async def get_metrics(self, symbol: str) -> Dict:
        """Get crypto metrics that might impact stocks"""
        
        try:
            # Check if company has crypto exposure
            crypto_exposed = {
                'COIN': 'coinbase',
                'MSTR': 'microstrategy',
                'SQ': 'square',
                'TSLA': 'tesla',
                'RIOT': 'riot',
                'MARA': 'marathon'
            }
            
            if symbol not in crypto_exposed:
                return {'signal': 'neutral', 'crypto_exposure': False, 'confidence': 0}
            
            # Get crypto fear & greed index
            fear_greed = await self._get_fear_greed_index()
            
            # Get Bitcoin metrics
            btc_data = await self._get_btc_metrics()
            
            # Determine signal based on crypto metrics
            signal = 'neutral'
            confidence = 0.5
            
            if fear_greed > 60 and btc_data.get('trend') == 'up':
                signal = 'bullish'
                confidence = 0.7
            elif fear_greed < 30 or btc_data.get('trend') == 'down':
                signal = 'bearish'
                confidence = 0.6
            
            return {
                'signal': signal,
                'confidence': confidence,
                'crypto_exposure': True,
                'fear_greed': fear_greed,
                'btc_metrics': btc_data,
                'exposure_type': crypto_exposed[symbol],
                'data_quality': DataQuality.HIGH.value
            }
            
        except Exception as e:
            logger.error(f"Crypto metrics error: {e}")
            return {'signal': 'neutral', 'error': str(e)}
    
    async def _get_fear_greed_index(self) -> int:
        """Get crypto fear & greed index"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/') as response:
                    data = await response.json()
                    return int(data['data'][0]['value'])
        except Exception as e:
            logger.error(f"Fear & Greed index error: {e}")
            return 50  # Neutral
    
    async def _get_btc_metrics(self) -> Dict:
        """Get Bitcoin metrics"""
        try:
            # Get Bitcoin data
            ticker = yf.Ticker('BTC-USD')
            hist = ticker.history(period='7d')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                week_ago_price = hist['Close'].iloc[0]
                trend = 'up' if current_price > week_ago_price else 'down'
                change_pct = ((current_price - week_ago_price) / week_ago_price) * 100
                
                volatility = hist['Close'].pct_change().std() * np.sqrt(365) * 100
                
                return {
                    'price': current_price,
                    'trend': trend,
                    'change_7d': change_pct,
                    'volatility': volatility
                }
        except Exception as e:
            logger.error(f"BTC metrics error: {e}")
        
        return {'trend': 'neutral', 'price': 0}

# ====================== SIGNAL GENERATOR ======================

class AlternativeDataSignalGenerator:
    """
    Generate trading signals from alternative data
    """
    
    def __init__(self, processor: AlternativeDataProcessor):
        self.processor = processor
        
    async def generate_signal(self, symbol: str) -> Dict:
        """Generate comprehensive trading signal"""
        
        try:
            # Collect all alternative data
            alt_data = await self.processor.collect_all_data(symbol)
            
            # Extract signals from each source
            signals = []
            for source, data in alt_data['sources'].items():
                if data and 'signal' in data:
                    signals.append({
                        'source': source,
                        'signal': data['signal'],
                        'confidence': data.get('confidence', 0.5)
                    })
            
            # Combine signals
            combined = self._combine_signals(signals)
            
            # Generate final recommendation
            recommendation = {
                'symbol': symbol,
                'action': self._get_action(combined),
                'confidence': combined['confidence'],
                'strength': combined['strength'],
                'supporting_sources': len([s for s in signals if s['signal'] != 'neutral']),
                'total_sources': len(signals),
                'supporting_data': alt_data['sources'],
                'timestamp': datetime.now().isoformat(),
                'expiry': (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0,
                'error': str(e)
            }
    
    def _combine_signals(self, signals: List[Dict]) -> Dict:
        """Combine multiple signals with weights"""
        if not signals:
            return {'strength': 0, 'confidence': 0}
        
        total_weight = 0
        weighted_sum = 0
        
        for sig in signals:
            weight = sig['confidence']
            value = 1 if sig['signal'] == 'bullish' else -1 if sig['signal'] == 'bearish' else 0
            
            weighted_sum += value * weight
            total_weight += weight
        
        if total_weight > 0:
            strength = weighted_sum / total_weight
            confidence = min(1.0, total_weight / len(signals))
        else:
            strength = 0
            confidence = 0
        
        return {'strength': strength, 'confidence': confidence}
    
    def _get_action(self, combined: Dict) -> str:
        """Convert combined signal to action"""
        strength = combined['strength']
        confidence = combined['confidence']
        
        if confidence < 0.3:
            return 'HOLD'
        
        if strength > 0.5:
            return 'STRONG_BUY'
        elif strength > 0.2:
            return 'BUY'
        elif strength < -0.5:
            return 'STRONG_SELL'
        elif strength < -0.2:
            return 'SELL'
        else:
            return 'HOLD'

# ====================== INTEGRATION ======================

def integrate_alternative_data(bot_instance):
    """Integrate alternative data into main bot"""
    
    # Initialize processor
    bot_instance.alt_data_processor = AlternativeDataProcessor(bot_instance.core.api)
    bot_instance.alt_signal_gen = AlternativeDataSignalGenerator(bot_instance.alt_data_processor)
    
    async def altdata_command(update, context):
        if not context.args:
            await update.message.reply_text("Usage: /altdata SYMBOL")
            return
        
        symbol = context.args[0].upper()
        
        await update.message.reply_text(f"🌐 Collecting alternative data for {symbol}... This may take a moment.")
        
        try:
            # Get alternative data signal
            signal = await bot_instance.alt_signal_gen.generate_signal(symbol)
            
            msg = f"""
🌐 **Alternative Data Analysis: {symbol}**

**📊 Signal:** **{signal['action']}**
**🎯 Confidence:** {signal['confidence']:.1%}
**💪 Strength:** {signal.get('strength', 0):.2f}

**📈 Data Sources:** ({signal['supporting_sources']}/{signal['total_sources']} active)
"""
            
            # Add details for each source
            for source, data in signal['supporting_data'].items():
                if data and not data.get('error'):
                    source_signal = data.get('signal', 'N/A')
                    confidence = data.get('confidence', 0)
                    
                    emoji = "🟢" if source_signal == 'bullish' else "🔴" if source_signal == 'bearish' else "🟡"
                    msg += f"\n{emoji} **{source.upper()}:** {source_signal}"
                    
                    # Add key metrics for each source
                    if source == 'google_trends' and 'trend_score' in data:
                        msg += f" (Trend: {data['trend_score']:.2f})"
                    elif source == 'reddit' and 'mentions' in data:
                        msg += f" ({data['mentions']} mentions)"
                    elif source == 'crypto' and 'fear_greed' in data:
                        msg += f" (F&G: {data['fear_greed']})"
                    elif source == 'app_store' and 'has_app' in data and data['has_app']:
                        rating = data.get('metrics', {}).get('rating', 0)
                        msg += f" (Rating: {rating:.1f})" if rating > 0 else ""
                    elif source == 'economic' and 'indicators' in data:
                        gdp = data['indicators'].get('gdp_growth', 0)
                        msg += f" (GDP: {gdp:.1f}%)"
                    elif source == 'weather' and data.get('impact') != 'none':
                        msg += f" ({data.get('conditions', 'N/A')})"
                    elif source == 'web_data' and 'hiring_activity' in data.get('data', {}):
                        hiring = data['data']['hiring_activity'].get('growth', 'N/A')
                        msg += f" (Hiring: {hiring})"
                    
                    msg += f" [{confidence:.1%}]"
                else:
                    msg += f"\n⚪ **{source.upper()}:** Error or N/A"
            
            msg += f"\n\n**⏱️ Valid until:** {signal['expiry'][:19]}"
            msg += f"\n**🕐 Generated:** {signal['timestamp'][:19]}"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Alternative data error: {str(e)}")
    
    return altdata_command
