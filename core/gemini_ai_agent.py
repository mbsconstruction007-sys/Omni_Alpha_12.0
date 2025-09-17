"""
STEP 14: GEMINI AI AGENT - Sentiment Analysis & News Integration
Advanced NLP using Google Gemini for market intelligence
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import hashlib
import time

import google.generativeai as genai
import feedparser
import pandas as pd

logger = logging.getLogger(__name__)

class GeminiAIAgent:
    """
    AI Agent powered by Google Gemini for sentiment analysis and news processing
    """
    
    def __init__(self):
        # Configure Gemini (using demo key for testing)
        self.api_key = os.getenv('GEMINI_API_KEY', 'demo_key')
        
        # For demo purposes, we'll simulate Gemini responses
        self.demo_mode = True  # Set to False when you have real API key
        
        if not self.demo_mode and self.api_key != 'demo_key':
            genai.configure(api_key=self.api_key)
            
            # Initialize model with optimal settings
            self.model = genai.GenerativeModel(
                'gemini-pro',
                generation_config={
                    'temperature': 0.3,  # Lower for more consistent analysis
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
        else:
            self.model = None
        
        # Cache system
        self.sentiment_cache = {}
        self.cache_duration = 15 * 60  # 15 minutes
        
        # Rate limiting
        self.request_times = deque(maxlen=60)
        self.max_requests_per_minute = 50
        
        # Analysis history
        self.analysis_history = deque(maxlen=1000)
        
    async def rate_limit_check(self):
        """Ensure we don't exceed Gemini rate limits"""
        current_time = time.time()
        
        # Clean old requests
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    def get_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def check_cache(self, key: str) -> Optional[Dict]:
        """Check if analysis is cached and valid"""
        if key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
        return None
    
    async def analyze_news_sentiment(self, symbol: str, articles: List[Dict]) -> Dict:
        """
        Comprehensive news sentiment analysis using Gemini
        """
        if not articles:
            return {'overall_sentiment': 0, 'confidence': 0, 'trading_signal': 'NEUTRAL'}
        
        # Create cache key
        content = json.dumps(articles[:10])  # Latest 10 articles
        cache_key = self.get_cache_key(content)
        
        # Check cache
        cached = self.check_cache(cache_key)
        if cached:
            return cached
        
        # Rate limit
        await self.rate_limit_check()
        
        try:
            if self.demo_mode:
                # Demo mode - simulate Gemini analysis
                analysis = self._simulate_news_analysis(symbol, articles)
            else:
                # Real Gemini analysis
                prompt = f"""
                You are a financial analyst AI. Analyze these news articles about {symbol} for trading decisions.
                
                Articles:
                {json.dumps(articles[:10], indent=2)}
                
                Provide a JSON response with EXACTLY this structure:
                {{
                    "overall_sentiment": <float between -1 and 1>,
                    "confidence": <integer 0-100>,
                    "key_events": [list of important events],
                    "trading_signal": "BUY" or "SELL" or "HOLD",
                    "risk_factors": [list of risks],
                    "catalysts": [positive factors],
                    "time_horizon": "IMMEDIATE" or "SHORT" or "MEDIUM" or "LONG",
                    "importance": <integer 1-10>,
                    "summary": "brief trading-focused summary"
                }}
                
                Consider:
                - Financial impact and materiality
                - Market reaction likelihood
                - Historical context
                - Multiple source confirmation
                - Sentiment momentum
                """
                
                response = self.model.generate_content(prompt)
                
                # Parse response
                try:
                    analysis = json.loads(response.text)
                except json.JSONDecodeError:
                    # Fallback parsing
                    analysis = self._parse_gemini_response(response.text)
            
            # Validate and normalize
            analysis = self._validate_analysis(analysis)
            
            # Cache result
            self.sentiment_cache[cache_key] = (analysis, time.time())
            
            # Store in history
            self.analysis_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'analysis': analysis
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return {
                'overall_sentiment': 0,
                'confidence': 0,
                'trading_signal': 'HOLD',
                'error': str(e)
            }
    
    def _simulate_news_analysis(self, symbol: str, articles: List[Dict]) -> Dict:
        """Simulate Gemini analysis for demo purposes"""
        
        # Simple keyword-based sentiment analysis
        positive_keywords = ['growth', 'profit', 'beat', 'strong', 'bullish', 'upgrade', 'buy']
        negative_keywords = ['loss', 'miss', 'weak', 'bearish', 'downgrade', 'sell', 'decline']
        
        sentiment_score = 0
        total_articles = len(articles)
        
        key_events = []
        risk_factors = []
        catalysts = []
        
        for article in articles[:10]:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            text = title + ' ' + summary
            
            # Count sentiment keywords
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            if positive_count > negative_count:
                sentiment_score += 0.3
                catalysts.append(article.get('title', '')[:50])
            elif negative_count > positive_count:
                sentiment_score -= 0.3
                risk_factors.append(article.get('title', '')[:50])
            
            # Extract key events
            if any(word in text for word in ['earnings', 'revenue', 'guidance']):
                key_events.append(f"Earnings-related: {article.get('title', '')[:60]}")
            elif any(word in text for word in ['acquisition', 'merger', 'deal']):
                key_events.append(f"M&A Activity: {article.get('title', '')[:60]}")
        
        # Normalize sentiment
        if total_articles > 0:
            sentiment_score = max(-1, min(1, sentiment_score / total_articles))
        
        # Generate trading signal
        if sentiment_score > 0.3:
            trading_signal = 'BUY'
            confidence = min(85, 50 + abs(sentiment_score) * 50)
        elif sentiment_score < -0.3:
            trading_signal = 'SELL'
            confidence = min(85, 50 + abs(sentiment_score) * 50)
        else:
            trading_signal = 'HOLD'
            confidence = 40
        
        return {
            'overall_sentiment': sentiment_score,
            'confidence': confidence,
            'key_events': key_events[:5],
            'trading_signal': trading_signal,
            'risk_factors': risk_factors[:3],
            'catalysts': catalysts[:3],
            'time_horizon': 'SHORT' if abs(sentiment_score) > 0.5 else 'MEDIUM',
            'importance': min(10, max(1, int(abs(sentiment_score) * 10) + 3)),
            'summary': f"Sentiment analysis of {total_articles} articles shows {trading_signal.lower()} signal"
        }
    
    async def analyze_social_sentiment(self, symbol: str, posts: List[str]) -> Dict:
        """
        Analyze social media sentiment for trading signals
        """
        if not posts:
            return {'retail_sentiment': 0, 'recommendation': 'NEUTRAL'}
        
        await self.rate_limit_check()
        
        try:
            if self.demo_mode:
                # Demo mode simulation
                return self._simulate_social_analysis(symbol, posts)
            
            prompt = f"""
            Analyze these social media posts about {symbol} for retail sentiment.
            
            Posts:
            {json.dumps(posts[:20], indent=2)}
            
            Return JSON with:
            {{
                "retail_sentiment": <float -1 to 1>,
                "sentiment_strength": "STRONG" or "MODERATE" or "WEAK",
                "bullish_count": <integer>,
                "bearish_count": <integer>,
                "trending_topics": [list of themes],
                "unusual_activity": <boolean>,
                "pump_risk": <float 0-1>,
                "influencer_mentions": <integer>,
                "emoji_sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
                "recommendation": "FOLLOW" or "FADE" or "IGNORE"
            }}
            
            Look for:
            - Coordinated pumping/dumping
            - Genuine enthusiasm vs hype
            - Technical analysis mentions
            - Risk warnings
            - FOMO indicators
            """
            
            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Social sentiment error: {e}")
            return {'retail_sentiment': 0, 'recommendation': 'IGNORE'}
    
    def _simulate_social_analysis(self, symbol: str, posts: List[str]) -> Dict:
        """Simulate social sentiment analysis"""
        
        bullish_keywords = ['moon', 'rocket', 'calls', 'bullish', 'buy', 'long']
        bearish_keywords = ['crash', 'puts', 'bearish', 'sell', 'short', 'dump']
        
        bullish_count = 0
        bearish_count = 0
        
        for post in posts:
            post_lower = post.lower()
            if any(word in post_lower for word in bullish_keywords):
                bullish_count += 1
            if any(word in post_lower for word in bearish_keywords):
                bearish_count += 1
        
        total_posts = len(posts)
        sentiment = (bullish_count - bearish_count) / total_posts if total_posts > 0 else 0
        
        return {
            'retail_sentiment': max(-1, min(1, sentiment)),
            'sentiment_strength': 'STRONG' if abs(sentiment) > 0.5 else 'MODERATE' if abs(sentiment) > 0.2 else 'WEAK',
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'trending_topics': ['earnings', 'technical_analysis'],
            'unusual_activity': bullish_count + bearish_count > total_posts * 0.8,
            'pump_risk': min(1, bullish_count / total_posts) if total_posts > 0 else 0,
            'influencer_mentions': 0,
            'emoji_sentiment': 'POSITIVE' if sentiment > 0 else 'NEGATIVE' if sentiment < 0 else 'NEUTRAL',
            'recommendation': 'FOLLOW' if abs(sentiment) > 0.3 else 'IGNORE'
        }
    
    async def identify_market_narratives(self, market_data: Dict) -> Dict:
        """
        Identify and track dominant market narratives
        """
        await self.rate_limit_check()
        
        try:
            if self.demo_mode:
                return self._simulate_narrative_analysis(market_data)
            
            prompt = f"""
            Identify current market narratives from this data.
            
            Market context:
            {json.dumps(market_data, indent=2)}
            
            Return JSON with:
            {{
                "dominant_themes": [list of top 3 themes],
                "sector_rotation": {{
                    "into_sectors": [sectors with inflows],
                    "out_of_sectors": [sectors with outflows]
                }},
                "risk_sentiment": "RISK_ON" or "RISK_OFF",
                "macro_drivers": [key macro factors],
                "sentiment_shift": "IMPROVING" or "DETERIORATING" or "STABLE",
                "contrarian_opportunity": <boolean>,
                "trend_strength": <1-10>,
                "regime": "BULL" or "BEAR" or "NEUTRAL",
                "key_risks": [major risk factors],
                "opportunities": [potential opportunities]
            }}
            """
            
            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Narrative analysis error: {e}")
            return {'dominant_themes': [], 'risk_sentiment': 'NEUTRAL'}
    
    def _simulate_narrative_analysis(self, market_data: Dict) -> Dict:
        """Simulate market narrative analysis"""
        
        # Simple simulation based on market data
        vix = market_data.get('vix', 15)
        spy_trend = market_data.get('spy_trend', 'neutral')
        
        if vix < 15:
            risk_sentiment = 'RISK_ON'
            themes = ['Growth Stocks', 'Tech Innovation', 'Economic Recovery']
        elif vix > 25:
            risk_sentiment = 'RISK_OFF'
            themes = ['Safe Haven Assets', 'Defensive Stocks', 'Market Volatility']
        else:
            risk_sentiment = 'NEUTRAL'
            themes = ['Mixed Signals', 'Sector Rotation', 'Earnings Focus']
        
        return {
            'dominant_themes': themes,
            'sector_rotation': {
                'into_sectors': ['Technology', 'Healthcare'],
                'out_of_sectors': ['Energy', 'Utilities']
            },
            'risk_sentiment': risk_sentiment,
            'macro_drivers': ['Fed Policy', 'Inflation Data', 'Employment'],
            'sentiment_shift': 'STABLE',
            'contrarian_opportunity': vix > 30,
            'trend_strength': 7 if spy_trend == 'up' else 3 if spy_trend == 'down' else 5,
            'regime': 'BULL' if spy_trend == 'up' else 'BEAR' if spy_trend == 'down' else 'NEUTRAL',
            'key_risks': ['Interest Rate Changes', 'Geopolitical Events'],
            'opportunities': ['AI Stocks', 'Cloud Computing']
        }
    
    def _validate_analysis(self, analysis: Dict) -> Dict:
        """Validate and normalize Gemini response"""
        
        # Ensure required fields
        defaults = {
            'overall_sentiment': 0,
            'confidence': 50,
            'trading_signal': 'HOLD',
            'risk_factors': [],
            'catalysts': [],
            'time_horizon': 'MEDIUM',
            'importance': 5,
            'summary': 'Analysis complete'
        }
        
        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
        
        # Clamp values
        analysis['overall_sentiment'] = max(-1, min(1, analysis.get('overall_sentiment', 0)))
        analysis['confidence'] = max(0, min(100, analysis.get('confidence', 50)))
        analysis['importance'] = max(1, min(10, analysis.get('importance', 5)))
        
        return analysis
    
    def _parse_gemini_response(self, text: str) -> Dict:
        """Fallback parser for non-JSON responses"""
        
        analysis = {
            'overall_sentiment': 0,
            'confidence': 50,
            'trading_signal': 'HOLD'
        }
        
        # Simple keyword parsing
        text_lower = text.lower()
        
        if 'buy' in text_lower:
            analysis['trading_signal'] = 'BUY'
            analysis['overall_sentiment'] = 0.5
        elif 'sell' in text_lower:
            analysis['trading_signal'] = 'SELL'
            analysis['overall_sentiment'] = -0.5
        
        if 'high confidence' in text_lower:
            analysis['confidence'] = 80
        elif 'low confidence' in text_lower:
            analysis['confidence'] = 20
        
        return analysis


class NewsDataCollector:
    """
    Collects news and social data for Gemini analysis
    """
    
    def __init__(self):
        self.news_sources = ['yahoo', 'benzinga']
        self.social_sources = ['reddit', 'stocktwits']
        
    async def fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news from multiple sources"""
        
        all_news = []
        
        # Yahoo Finance RSS
        if 'yahoo' in self.news_sources:
            yahoo_news = await self._fetch_yahoo_news(symbol)
            all_news.extend(yahoo_news)
        
        # Sort by timestamp
        all_news.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        return all_news[:50]  # Return top 50 most recent
    
    async def _fetch_yahoo_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance RSS"""
        
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    content = await response.text()
            
            feed = feedparser.parse(content)
            
            news = []
            for entry in feed.entries[:20]:
                news.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': 'Yahoo Finance'
                })
            
            return news
            
        except Exception as e:
            logger.error(f"Yahoo news fetch error: {e}")
            return []
    
    async def fetch_social_sentiment(self, symbol: str) -> List[str]:
        """Fetch social media posts (simplified version)"""
        
        # In production, you'd use Reddit API, Twitter API, etc.
        # This is a demo structure
        
        sample_posts = [
            f"${symbol} earnings beat expectations! 🚀",
            f"Loading up on ${symbol} calls for next week",
            f"${symbol} looking bearish on the charts",
            f"Great quarterly results from ${symbol}",
            f"${symbol} breaking resistance, going long",
            f"Worried about ${symbol} guidance",
            f"${symbol} to $300 by year end",
            f"Selling my ${symbol} position today",
            f"${symbol} institutional buying detected",
            f"${symbol} options flow looking bullish"
        ]
        
        return sample_posts


class SentimentTradingSignals:
    """
    Generates trading signals from Gemini analysis
    """
    
    def __init__(self, gemini_agent: GeminiAIAgent, news_collector: NewsDataCollector):
        self.gemini = gemini_agent
        self.collector = news_collector
        self.signal_history = deque(maxlen=100)
        
    async def generate_signal(self, symbol: str) -> Dict:
        """
        Generate comprehensive trading signal from sentiment
        """
        
        try:
            # Collect data
            news = await self.collector.fetch_news(symbol)
            social = await self.collector.fetch_social_sentiment(symbol)
            
            # Get Gemini analysis
            news_sentiment = await self.gemini.analyze_news_sentiment(symbol, news)
            social_sentiment = await self.gemini.analyze_social_sentiment(symbol, social)
            
            # Combine signals
            combined_sentiment = (
                news_sentiment.get('overall_sentiment', 0) * 0.7 +
                social_sentiment.get('retail_sentiment', 0) * 0.3
            )
            
            # Calculate confidence
            news_confidence = news_sentiment.get('confidence', 50)
            social_confidence = 40 if social_sentiment.get('recommendation') == 'FOLLOW' else 20
            
            confidence = (news_confidence * 0.6 + social_confidence * 0.4)
            
            # Determine signal
            if combined_sentiment > 0.3 and confidence > 60:
                signal = 'BUY'
            elif combined_sentiment < -0.3 and confidence > 60:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Risk assessment
            risk_factors = news_sentiment.get('risk_factors', [])
            risk_level = 'HIGH' if len(risk_factors) > 2 else 'MEDIUM' if len(risk_factors) > 0 else 'LOW'
            
            result = {
                'symbol': symbol,
                'signal': signal,
                'sentiment_score': combined_sentiment,
                'confidence': confidence,
                'news_sentiment': news_sentiment.get('overall_sentiment', 0),
                'social_sentiment': social_sentiment.get('retail_sentiment', 0),
                'risk_level': risk_level,
                'time_horizon': news_sentiment.get('time_horizon', 'MEDIUM'),
                'key_factors': {
                    'news': news_sentiment.get('summary', ''),
                    'social': social_sentiment.get('trending_topics', []),
                    'risks': news_sentiment.get('risk_factors', []),
                    'catalysts': news_sentiment.get('catalysts', [])
                },
                'news_count': len(news),
                'social_posts_count': len(social),
                'timestamp': datetime.now().isoformat()
            }
            
            self.signal_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'sentiment_score': 0,
                'confidence': 0,
                'error': str(e)
            }


# Integration with main bot
def integrate_gemini_agent(bot_instance):
    """Add Gemini AI Agent to OmniAlphaTelegramBot"""
    
    # Initialize Gemini components
    bot_instance.gemini_agent = GeminiAIAgent()
    bot_instance.news_collector = NewsDataCollector()
    bot_instance.sentiment_signals = SentimentTradingSignals(
        bot_instance.gemini_agent,
        bot_instance.news_collector
    )
    
    # Add command handlers
    async def sentiment_command(update, context):
        if not context.args:
            await update.message.reply_text("Usage: /sentiment SYMBOL")
            return
        
        symbol = context.args[0].upper()
        
        await update.message.reply_text(f"🤖 Analyzing sentiment for {symbol} with Gemini AI... This may take a moment.")
        
        try:
            # Get sentiment signal
            signal = await bot_instance.sentiment_signals.generate_signal(symbol)
            
            msg = f"""
🤖 **Gemini AI Sentiment Analysis: {symbol}**

**📊 Trading Signal:** **{signal['signal']}**
**🎯 Confidence:** {signal['confidence']:.1f}%

**📈 Sentiment Scores:**
• 📰 News: {signal['news_sentiment']:.2f}
• 💬 Social: {signal['social_sentiment']:.2f}
• 📊 Combined: {signal['sentiment_score']:.2f}

**⚠️ Risk Level:** {signal['risk_level']}
**⏱️ Time Horizon:** {signal['time_horizon']}

**📰 News Analysis:**
{signal['key_factors']['news'][:200]}...

**⚠️ Risk Factors:**
{chr(10).join('• ' + risk for risk in signal['key_factors']['risks'][:3]) if signal['key_factors']['risks'] else '• None identified'}

**🚀 Catalysts:**
{chr(10).join('• ' + catalyst for catalyst in signal['key_factors']['catalysts'][:3]) if signal['key_factors']['catalysts'] else '• None identified'}

**📊 Data Sources:**
• News Articles: {signal['news_count']}
• Social Posts: {signal['social_posts_count']}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing sentiment: {str(e)}")
    
    async def news_command(update, context):
        if not context.args:
            await update.message.reply_text("Usage: /news SYMBOL")
            return
        
        symbol = context.args[0].upper()
        
        await update.message.reply_text(f"📰 Fetching latest news for {symbol}...")
        
        try:
            # Fetch latest news
            news = await bot_instance.news_collector.fetch_news(symbol)
            
            if news:
                msg = f"📰 **Latest News for {symbol}:**\n\n"
                for i, article in enumerate(news[:5], 1):
                    msg += f"**{i}.** {article['title']}\n"
                    if 'published' in article and article['published']:
                        msg += f"   📅 {article['published'][:16]}\n"
                    if 'summary' in article and article['summary']:
                        msg += f"   📝 {article['summary'][:100]}...\n"
                    msg += "\n"
                
                msg += f"📊 Total articles found: {len(news)}"
            else:
                msg = f"❌ No recent news found for {symbol}"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error fetching news: {str(e)}")
    
    async def narrative_command(update, context):
        await update.message.reply_text("🌐 Analyzing market narratives with Gemini AI...")
        
        try:
            # Get market data for narrative analysis
            market_data = {
                'vix': 15.5,
                'spy_trend': 'up',
                'volume': 'above_average',
                'sectors': ['tech', 'healthcare'],
                'timestamp': datetime.now().isoformat()
            }
            
            narratives = await bot_instance.gemini_agent.identify_market_narratives(market_data)
            
            msg = f"""
🌐 **Market Narratives (Powered by Gemini AI)**

**🎯 Dominant Themes:**
{chr(10).join('• ' + theme for theme in narratives.get('dominant_themes', []))}

**📊 Risk Sentiment:** {narratives.get('risk_sentiment', 'NEUTRAL')}
**📈 Market Regime:** {narratives.get('regime', 'NEUTRAL')}
**💪 Trend Strength:** {narratives.get('trend_strength', 5)}/10

**🔄 Sector Rotation:**
• Into: {', '.join(narratives.get('sector_rotation', {}).get('into_sectors', []))}
• Out: {', '.join(narratives.get('sector_rotation', {}).get('out_of_sectors', []))}

**🌍 Macro Drivers:**
{chr(10).join('• ' + driver for driver in narratives.get('macro_drivers', []))}

**⚠️ Key Risks:**
{chr(10).join('• ' + risk for risk in narratives.get('key_risks', []))}

**🚀 Opportunities:**
{chr(10).join('• ' + opp for opp in narratives.get('opportunities', []))}

**📊 Sentiment Shift:** {narratives.get('sentiment_shift', 'STABLE')}
**🔄 Contrarian Opportunity:** {'Yes' if narratives.get('contrarian_opportunity') else 'No'}
            """
            
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error analyzing narratives: {str(e)}")
    
    return sentiment_command, news_command, narrative_command
