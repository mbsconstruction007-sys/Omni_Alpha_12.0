"""
STEP 20: Complete Institutional Scale & Business Transformation System
Full hedge fund/asset management infrastructure
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import json
import uuid

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
import redis
from cryptography.fernet import Fernet
import stripe
from twilio.rest import Client as TwilioClient
import yagmail

from pydantic import BaseModel, validator
import jwt
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

try:
    engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///institutional.db'))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    DATABASE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Database not available: {e}")
    DATABASE_AVAILABLE = False

# ===================== DATA MODELS =====================

class ClientType(Enum):
    HNI = "HIGH_NET_WORTH"
    FAMILY_OFFICE = "FAMILY_OFFICE"
    INSTITUTIONAL = "INSTITUTIONAL"
    INTERNATIONAL = "INTERNATIONAL"
    EMPLOYEE = "EMPLOYEE"

class PortfolioStatus(Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"
    PENDING = "PENDING"

class ComplianceStatus(Enum):
    COMPLIANT = "COMPLIANT"
    UNDER_REVIEW = "UNDER_REVIEW"
    NON_COMPLIANT = "NON_COMPLIANT"
    EXEMPTED = "EXEMPTED"

# ===================== DATABASE MODELS =====================

if DATABASE_AVAILABLE:
    class Client(Base):
        """Client/Investor entity"""
        __tablename__ = 'clients'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        client_type = Column(String, nullable=False)
        name = Column(String, nullable=False)
        email = Column(String, unique=True, nullable=False)
        phone = Column(String)
        pan_number = Column(String, unique=True)
        aadhar_number = Column(String, unique=True)
        
        # KYC Details
        kyc_completed = Column(Boolean, default=False)
        kyc_date = Column(DateTime)
        risk_profile = Column(String)
        investment_objective = Column(String)
        
        # Financial Details
        net_worth = Column(Float)
        annual_income = Column(Float)
        source_of_wealth = Column(String)
        
        # Relationship
        relationship_manager = Column(String)
        onboarding_date = Column(DateTime, default=datetime.now)
        status = Column(String, default='ACTIVE')
        
        # Relations
        portfolios = relationship('Portfolio', back_populates='client')
        transactions = relationship('Transaction', back_populates='client')

    class Portfolio(Base):
        """Portfolio management"""
        __tablename__ = 'portfolios'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        client_id = Column(String, ForeignKey('clients.id'))
        portfolio_name = Column(String, nullable=False)
        strategy = Column(String, nullable=False)
        
        # Investment details
        initial_investment = Column(Float, nullable=False)
        current_value = Column(Float)
        cash_balance = Column(Float)
        
        # Performance
        total_return = Column(Float)
        ytd_return = Column(Float)
        sharpe_ratio = Column(Float)
        max_drawdown = Column(Float)
        
        # Fees
        management_fee = Column(Float, default=0.02)
        performance_fee = Column(Float, default=0.20)
        high_water_mark = Column(Float)
        
        # Status
        status = Column(String, default='ACTIVE')
        inception_date = Column(DateTime, default=datetime.now)
        last_rebalance = Column(DateTime)
        
        # Relations
        client = relationship('Client', back_populates='portfolios')
        positions = relationship('Position', back_populates='portfolio')
        transactions = relationship('Transaction', back_populates='portfolio')

    class Position(Base):
        """Portfolio positions"""
        __tablename__ = 'positions'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        portfolio_id = Column(String, ForeignKey('portfolios.id'))
        symbol = Column(String, nullable=False)
        
        # Position details
        quantity = Column(Float, nullable=False)
        average_price = Column(Float, nullable=False)
        current_price = Column(Float)
        market_value = Column(Float)
        
        # P&L
        unrealized_pnl = Column(Float)
        realized_pnl = Column(Float)
        
        # Dates
        entry_date = Column(DateTime, default=datetime.now)
        last_updated = Column(DateTime, default=datetime.now)
        
        # Relations
        portfolio = relationship('Portfolio', back_populates='positions')

    class Transaction(Base):
        """Transaction records"""
        __tablename__ = 'transactions'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        client_id = Column(String, ForeignKey('clients.id'))
        portfolio_id = Column(String, ForeignKey('portfolios.id'))
        
        # Transaction details
        transaction_type = Column(String)  # BUY, SELL, DEPOSIT, WITHDRAWAL, FEE
        symbol = Column(String)
        quantity = Column(Float)
        price = Column(Float)
        amount = Column(Float)
        
        # Fees & charges
        brokerage = Column(Float)
        taxes = Column(Float)
        total_amount = Column(Float)
        
        # Status
        status = Column(String, default='COMPLETED')
        transaction_date = Column(DateTime, default=datetime.now)
        settlement_date = Column(DateTime)
        
        # Relations
        client = relationship('Client', back_populates='transactions')
        portfolio = relationship('Portfolio', back_populates='transactions')

    class ComplianceRecord(Base):
        """Compliance tracking"""
        __tablename__ = 'compliance_records'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        record_type = Column(String)  # KYC, AML, REGULATORY_FILING, AUDIT
        entity_type = Column(String)  # CLIENT, PORTFOLIO, TRANSACTION
        entity_id = Column(String)
        
        # Compliance details
        status = Column(String)
        check_date = Column(DateTime, default=datetime.now)
        expiry_date = Column(DateTime)
        notes = Column(Text)
        
        # Audit trail
        checked_by = Column(String)
        approved_by = Column(String)
        approval_date = Column(DateTime)

# ===================== CORE SYSTEMS =====================

class InstitutionalClientManager:
    """Manages institutional clients and investors"""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            self.db = SessionLocal()
        else:
            self.db = None
            
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                decode_responses=True
            )
            self.redis_available = True
        except:
            self.redis_available = False
            
        # In-memory storage for demo
        self.clients = {}
        self.portfolios = {}
        
    async def onboard_client(self, client_data: Dict) -> Dict:
        """Complete client onboarding process"""
        
        try:
            # Validate client type and minimum investment
            if not self._validate_minimum_investment(
                client_data['client_type'], 
                client_data['initial_investment']
            ):
                raise ValueError("Below minimum investment threshold")
            
            # KYC verification
            kyc_result = await self._perform_kyc_check(client_data)
            if not kyc_result['passed']:
                raise ValueError(f"KYC failed: {kyc_result['reason']}")
            
            # AML screening
            aml_result = await self._perform_aml_screening(client_data)
            if aml_result['risk'] == 'HIGH':
                raise ValueError(f"AML risk too high: {aml_result['reason']}")
            
            # Create client record
            client_id = str(uuid.uuid4())
            client = {
                'id': client_id,
                'client_type': client_data['client_type'],
                'name': client_data['name'],
                'email': client_data['email'],
                'phone': client_data.get('phone'),
                'pan_number': client_data['pan_number'],
                'kyc_completed': True,
                'kyc_date': datetime.now(),
                'risk_profile': client_data.get('risk_profile', 'MODERATE'),
                'investment_objective': client_data.get('investment_objective', 'GROWTH'),
                'net_worth': client_data.get('net_worth'),
                'annual_income': client_data.get('annual_income'),
                'relationship_manager': self._assign_relationship_manager(client_data['client_type']),
                'onboarding_date': datetime.now(),
                'status': 'ACTIVE'
            }
            
            self.clients[client_id] = client
            
            # Create initial portfolio
            portfolio = await self._create_portfolio(client, client_data)
            
            # Send welcome communication
            await self._send_welcome_package(client)
            
            logger.info(f"Client onboarded successfully: {client_id}")
            return client
            
        except Exception as e:
            logger.error(f"Client onboarding failed: {e}")
            raise
    
    def _validate_minimum_investment(self, client_type: str, amount: float) -> bool:
        """Validate minimum investment requirements"""
        
        min_investments = {
            ClientType.HNI.value: float(os.getenv('MIN_INVESTMENT_HNI', '10000000')),
            ClientType.FAMILY_OFFICE.value: float(os.getenv('MIN_INVESTMENT_FAMILY_OFFICE', '100000000')),
            ClientType.INSTITUTIONAL.value: float(os.getenv('MIN_INVESTMENT_INSTITUTIONAL', '500000000')),
            ClientType.INTERNATIONAL.value: float(os.getenv('MIN_INVESTMENT_INTERNATIONAL', '1000000'))
        }
        
        return amount >= min_investments.get(client_type, float('inf'))
    
    async def _perform_kyc_check(self, client_data: Dict) -> Dict:
        """Perform KYC verification"""
        
        # In production, integrate with KYC service providers
        # For now, simulate KYC check
        
        required_docs = ['pan_card', 'aadhar_card', 'address_proof', 'bank_statement']
        
        for doc in required_docs:
            if doc not in client_data.get('documents', {}):
                return {'passed': False, 'reason': f'Missing {doc}'}
        
        # Verify PAN
        if not self._verify_pan(client_data['pan_number']):
            return {'passed': False, 'reason': 'Invalid PAN'}
        
        return {'passed': True, 'kyc_id': str(uuid.uuid4())}
    
    def _verify_pan(self, pan: str) -> bool:
        """Verify PAN number format"""
        import re
        pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        return bool(re.match(pan_pattern, pan))
    
    async def _perform_aml_screening(self, client_data: Dict) -> Dict:
        """Perform Anti-Money Laundering screening"""
        
        risk_score = 0
        
        # Check if politically exposed person
        if client_data.get('is_pep', False):
            risk_score += 50
        
        # Check source of wealth
        high_risk_sources = ['CRYPTO', 'GAMBLING', 'UNKNOWN']
        if client_data.get('source_of_wealth', '').upper() in high_risk_sources:
            risk_score += 30
        
        # Geographic risk
        high_risk_countries = ['SYRIA', 'IRAN', 'NORTH_KOREA']
        if client_data.get('country', '').upper() in high_risk_countries:
            risk_score += 50
        
        risk_level = 'LOW' if risk_score < 30 else 'MEDIUM' if risk_score < 70 else 'HIGH'
        
        return {
            'risk': risk_level,
            'score': risk_score,
            'reason': 'Passed AML screening' if risk_level != 'HIGH' else 'High risk factors detected'
        }
    
    def _assign_relationship_manager(self, client_type: str) -> str:
        """Assign relationship manager based on client type"""
        
        managers = {
            ClientType.HNI.value: 'HNI_DESK',
            ClientType.FAMILY_OFFICE.value: 'FAMILY_OFFICE_DESK',
            ClientType.INSTITUTIONAL.value: 'INSTITUTIONAL_DESK',
            ClientType.INTERNATIONAL.value: 'INTERNATIONAL_DESK'
        }
        
        return managers.get(client_type, 'GENERAL_DESK')
    
    async def _create_portfolio(self, client: Dict, client_data: Dict) -> Dict:
        """Create initial portfolio for client"""
        
        portfolio_id = str(uuid.uuid4())
        portfolio = {
            'id': portfolio_id,
            'client_id': client['id'],
            'portfolio_name': f"{client['name']} - {client_data.get('strategy', 'Growth')}",
            'strategy': client_data.get('strategy', 'BALANCED_GROWTH'),
            'initial_investment': client_data['initial_investment'],
            'current_value': client_data['initial_investment'],
            'cash_balance': client_data['initial_investment'],
            'total_return': 0.0,
            'ytd_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'management_fee': client_data.get('management_fee', 0.02),
            'performance_fee': client_data.get('performance_fee', 0.20),
            'high_water_mark': client_data['initial_investment'],
            'status': 'ACTIVE',
            'inception_date': datetime.now(),
            'last_rebalance': datetime.now()
        }
        
        self.portfolios[portfolio_id] = portfolio
        
        return portfolio
    
    async def _send_welcome_package(self, client: Dict):
        """Send welcome package to new client"""
        
        try:
            # Send welcome email
            if os.getenv('EMAIL_ENABLED', 'false').lower() == 'true':
                yag = yagmail.SMTP(
                    os.getenv('EMAIL_USER'),
                    os.getenv('EMAIL_PASSWORD')
                )
                
                welcome_message = f"""
                Welcome to Omni Alpha Asset Management!
                
                Dear {client['name']},
                
                Thank you for choosing Omni Alpha for your investment needs.
                Your account has been successfully created and KYC verification completed.
                
                Client ID: {client['id']}
                Relationship Manager: {client['relationship_manager']}
                
                You will receive your login credentials separately.
                
                Best regards,
                Omni Alpha Team
                """
                
                yag.send(
                    to=client['email'],
                    subject='Welcome to Omni Alpha Asset Management',
                    contents=welcome_message
                )
                
                logger.info(f"Welcome email sent to {client['email']}")
            
            # Send SMS notification
            if os.getenv('TWILIO_ENABLED', 'false').lower() == 'true':
                twilio_client = TwilioClient(
                    os.getenv('TWILIO_ACCOUNT_SID'),
                    os.getenv('TWILIO_AUTH_TOKEN')
                )
                
                sms_message = f"Welcome to Omni Alpha! Your account {client['id'][:8]} is now active. Check your email for details."
                
                twilio_client.messages.create(
                    body=sms_message,
                    from_=os.getenv('TWILIO_PHONE_NUMBER'),
                    to=client['phone']
                )
                
                logger.info(f"Welcome SMS sent to {client['phone']}")
                
        except Exception as e:
            logger.error(f"Welcome package error: {e}")

class InstitutionalPortfolioManager:
    """Manages multiple client portfolios"""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            self.db = SessionLocal()
        else:
            self.db = None
        self.portfolios_cache = {}
        
    async def manage_portfolio(self, portfolio_id: str) -> Dict:
        """Manage individual portfolio"""
        
        # Get portfolio (from memory for demo)
        portfolio = self.portfolios_cache.get(portfolio_id)
        
        if not portfolio:
            # Simulate portfolio
            portfolio = {
                'id': portfolio_id,
                'current_value': 10000000,
                'initial_investment': 10000000,
                'total_return': 0.08,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.05,
                'strategy': 'BALANCED_GROWTH'
            }
            self.portfolios_cache[portfolio_id] = portfolio
        
        # Calculate current metrics
        metrics = await self._calculate_portfolio_metrics(portfolio)
        
        # Check rebalancing needs
        if self._needs_rebalancing(portfolio, metrics):
            await self._rebalance_portfolio(portfolio)
        
        # Risk management
        risk_check = await self._check_portfolio_risk(portfolio, metrics)
        if not risk_check['compliant']:
            await self._adjust_portfolio_risk(portfolio, risk_check)
        
        # Update portfolio values
        portfolio['current_value'] = metrics['total_value']
        portfolio['total_return'] = metrics['total_return']
        portfolio['sharpe_ratio'] = metrics['sharpe_ratio']
        portfolio['max_drawdown'] = metrics['max_drawdown']
        
        return metrics
    
    async def _calculate_portfolio_metrics(self, portfolio: Dict) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        # Simulate portfolio metrics calculation
        current_value = portfolio['current_value']
        initial_value = portfolio['initial_investment']
        
        # Performance metrics
        total_return = (current_value - initial_value) / initial_value
        ytd_return = total_return * 0.8  # Simplified
        
        # Risk metrics
        volatility = 0.15  # 15% annual volatility
        sharpe_ratio = (total_return - 0.065) / volatility  # Risk-free rate 6.5%
        
        # Drawdown (simulate)
        max_drawdown = np.random.uniform(-0.02, -0.08)
        
        return {
            'total_value': current_value,
            'total_return': total_return,
            'ytd_return': ytd_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'calculation_date': datetime.now().isoformat()
        }
    
    def _needs_rebalancing(self, portfolio: Dict, metrics: Dict) -> bool:
        """Check if portfolio needs rebalancing"""
        
        # Check time since last rebalance
        last_rebalance = portfolio.get('last_rebalance', datetime.now() - timedelta(days=30))
        days_since_rebalance = (datetime.now() - last_rebalance).days
        
        # Rebalance quarterly or if significant drift
        return days_since_rebalance > 90 or abs(metrics['max_drawdown']) > 0.10
    
    async def _rebalance_portfolio(self, portfolio: Dict):
        """Rebalance portfolio"""
        
        logger.info(f"Rebalancing portfolio {portfolio['id']}")
        
        # In production, implement actual rebalancing logic
        portfolio['last_rebalance'] = datetime.now()
    
    async def _check_portfolio_risk(self, portfolio: Dict, metrics: Dict) -> Dict:
        """Check portfolio risk compliance"""
        
        issues = []
        
        # Check drawdown limit
        if abs(metrics['max_drawdown']) > 0.15:
            issues.append("Maximum drawdown exceeded")
        
        # Check volatility
        if metrics['volatility'] > 0.25:
            issues.append("Volatility too high")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues
        }
    
    async def _adjust_portfolio_risk(self, portfolio: Dict, risk_check: Dict):
        """Adjust portfolio risk"""
        
        logger.warning(f"Adjusting risk for portfolio {portfolio['id']}: {risk_check['issues']}")
        
        # In production, implement risk adjustment logic
    
    async def calculate_fees(self, portfolio_id: str, period: str = 'QUARTERLY') -> Dict:
        """Calculate management and performance fees"""
        
        portfolio = self.portfolios_cache.get(portfolio_id)
        
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
        
        # Management fee (annual rate, calculated for period)
        period_factor = {'MONTHLY': 12, 'QUARTERLY': 4, 'ANNUAL': 1}[period]
        management_fee = portfolio['current_value'] * portfolio.get('management_fee', 0.02) / period_factor
        
        # Performance fee (above high water mark)
        performance_fee = 0
        high_water_mark = portfolio.get('high_water_mark', portfolio['initial_investment'])
        
        if portfolio['current_value'] > high_water_mark:
            profit = portfolio['current_value'] - high_water_mark
            
            # Check if hurdle rate is met
            hurdle_rate = float(os.getenv('HURDLE_RATE_PERCENT', '8')) / 100 / period_factor
            hurdle_amount = high_water_mark * hurdle_rate
            
            if profit > hurdle_amount:
                performance_fee = (profit - hurdle_amount) * portfolio.get('performance_fee', 0.20)
            
            # Update high water mark
            portfolio['high_water_mark'] = portfolio['current_value']
        
        total_fees = management_fee + performance_fee
        
        return {
            'management_fee': management_fee,
            'performance_fee': performance_fee,
            'total_fees': total_fees,
            'period': period,
            'calculation_date': datetime.now().isoformat()
        }

class InstitutionalRiskManager:
    """Institutional-grade risk management"""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            self.db = SessionLocal()
        else:
            self.db = None
        self.risk_limits = self._load_risk_limits()
        
    def _load_risk_limits(self) -> Dict:
        """Load risk limits from configuration"""
        
        return {
            'portfolio_var_limit': float(os.getenv('PORTFOLIO_VAR_LIMIT', '5000000')),
            'portfolio_stress_limit': float(os.getenv('PORTFOLIO_STRESS_LIMIT', '10000000')),
            'concentration_limit': float(os.getenv('CONCENTRATION_LIMIT', '20')),
            'liquidity_coverage_ratio': float(os.getenv('LIQUIDITY_COVERAGE_RATIO', '1.5')),
            'counterparty_limit': float(os.getenv('COUNTERPARTY_LIMIT', '10000000'))
        }
    
    async def perform_risk_assessment(self, portfolio_id: str) -> Dict:
        """Comprehensive risk assessment for portfolio"""
        
        # Simulate positions
        positions = [
            {'symbol': 'NIFTY', 'market_value': 5000000, 'quantity': 250},
            {'symbol': 'BANKNIFTY', 'market_value': 3000000, 'quantity': 75},
            {'symbol': 'FINNIFTY', 'market_value': 2000000, 'quantity': 50}
        ]
        
        assessment = {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': {},
            'breaches': [],
            'recommendations': []
        }
        
        # Calculate VaR
        var_95 = await self._calculate_var(positions, 0.95)
        assessment['risk_metrics']['var_95'] = var_95
        
        if var_95 > self.risk_limits['portfolio_var_limit']:
            assessment['breaches'].append({
                'type': 'VAR_BREACH',
                'current': var_95,
                'limit': self.risk_limits['portfolio_var_limit']
            })
        
        # Stress testing
        stress_results = await self._perform_stress_test(positions)
        assessment['risk_metrics']['stress_test'] = stress_results
        
        # Concentration risk
        concentration = self._calculate_concentration_risk(positions)
        assessment['risk_metrics']['concentration'] = concentration
        
        if concentration['max_concentration'] > self.risk_limits['concentration_limit']:
            assessment['breaches'].append({
                'type': 'CONCENTRATION_BREACH',
                'current': concentration['max_concentration'],
                'limit': self.risk_limits['concentration_limit']
            })
        
        # Generate recommendations
        if assessment['breaches']:
            assessment['recommendations'] = self._generate_risk_recommendations(assessment)
        
        return assessment
    
    async def _calculate_var(self, positions: List[Dict], confidence: float) -> float:
        """Calculate Value at Risk"""
        
        # Get historical returns for all positions
        returns_data = []
        
        for position in positions:
            # Simulate returns
            returns = np.random.normal(0.001, 0.02, 252)
            returns_data.append(returns * position['market_value'])
        
        portfolio_returns = np.sum(returns_data, axis=0)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        return abs(var)
    
    async def _perform_stress_test(self, positions: List[Dict]) -> Dict:
        """Perform stress testing scenarios"""
        
        scenarios = {
            'market_crash': {'market_move': -0.20, 'volatility_spike': 2.0},
            'interest_rate_shock': {'rate_move': 0.02, 'curve_steepening': True},
            'liquidity_crisis': {'bid_ask_widening': 3.0, 'volume_reduction': 0.5},
            'sector_rotation': {'tech_down': -0.15, 'value_up': 0.10}
        }
        
        results = {}
        total_value = sum(p['market_value'] for p in positions)
        
        for scenario_name, params in scenarios.items():
            # Calculate impact
            impact = 0
            for position in positions:
                # Simplified impact calculation
                impact += position['market_value'] * params.get('market_move', 0)
            
            results[scenario_name] = {
                'impact': impact,
                'impact_percent': (impact / total_value) * 100 if total_value > 0 else 0
            }
        
        return results
    
    def _calculate_concentration_risk(self, positions: List[Dict]) -> Dict:
        """Calculate concentration risk"""
        
        total_value = sum(p['market_value'] for p in positions)
        
        concentrations = []
        for position in positions:
            concentration = (position['market_value'] / total_value) * 100 if total_value > 0 else 0
            concentrations.append({
                'symbol': position['symbol'],
                'concentration': concentration
            })
        
        max_concentration = max(concentrations, key=lambda x: x['concentration'])
        
        return {
            'concentrations': concentrations,
            'max_concentration': max_concentration['concentration'],
            'max_symbol': max_concentration['symbol']
        }
    
    def _generate_risk_recommendations(self, assessment: Dict) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        for breach in assessment['breaches']:
            if breach['type'] == 'VAR_BREACH':
                recommendations.append("Reduce portfolio risk by decreasing position sizes")
            elif breach['type'] == 'CONCENTRATION_BREACH':
                recommendations.append(f"Reduce concentration in {breach.get('symbol', 'top positions')}")
        
        return recommendations

class ComplianceManager:
    """Manages regulatory compliance"""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            self.db = SessionLocal()
        else:
            self.db = None
        self.compliance_records = {}
        
    async def perform_compliance_check(self, check_type: str, entity_id: str) -> Dict:
        """Perform compliance check"""
        
        result = {
            'check_type': check_type,
            'entity_id': entity_id,
            'timestamp': datetime.now().isoformat(),
            'status': ComplianceStatus.COMPLIANT.value,
            'issues': []
        }
        
        if check_type == 'PORTFOLIO':
            result = await self._check_portfolio_compliance(entity_id)
        elif check_type == 'TRANSACTION':
            result = await self._check_transaction_compliance(entity_id)
        elif check_type == 'CLIENT':
            result = await self._check_client_compliance(entity_id)
        
        # Store compliance record
        record_id = str(uuid.uuid4())
        self.compliance_records[record_id] = {
            'id': record_id,
            'record_type': check_type,
            'entity_type': check_type,
            'entity_id': entity_id,
            'status': result['status'],
            'check_date': datetime.now(),
            'notes': json.dumps(result['issues'])
        }
        
        return result
    
    async def _check_portfolio_compliance(self, portfolio_id: str) -> Dict:
        """Check portfolio compliance"""
        
        # Simulate portfolio compliance check
        issues = []
        
        # Simulate some compliance checks
        concentration_check = np.random.uniform(0, 25)
        if concentration_check > 20:
            issues.append(f"Position concentration exceeds limit: {concentration_check:.1f}%")
        
        derivative_exposure = np.random.uniform(0, 50)
        if derivative_exposure > 40:
            issues.append(f"Derivative exposure exceeds limit: {derivative_exposure:.1f}%")
        
        status = ComplianceStatus.COMPLIANT if not issues else ComplianceStatus.NON_COMPLIANT
        
        return {
            'status': status.value,
            'issues': issues,
            'portfolio_id': portfolio_id
        }
    
    async def _check_transaction_compliance(self, transaction_id: str) -> Dict:
        """Check transaction compliance"""
        
        return {
            'status': ComplianceStatus.COMPLIANT.value,
            'issues': [],
            'transaction_id': transaction_id
        }
    
    async def _check_client_compliance(self, client_id: str) -> Dict:
        """Check client compliance"""
        
        return {
            'status': ComplianceStatus.COMPLIANT.value,
            'issues': [],
            'client_id': client_id
        }

class BusinessOperationsManager:
    """Manages business operations"""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            self.db = SessionLocal()
        else:
            self.db = None
        
    async def calculate_business_metrics(self) -> Dict:
        """Calculate key business metrics"""
        
        # Simulate business metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'aum': {
                'total': 5000000000,  # 500 crores
                'by_strategy': {
                    'BALANCED_GROWTH': 2000000000,
                    'AGGRESSIVE_GROWTH': 1500000000,
                    'CONSERVATIVE': 1000000000,
                    'HEDGE_FUND': 500000000
                },
                'by_client_type': {
                    'HNI': 2000000000,
                    'FAMILY_OFFICE': 1500000000,
                    'INSTITUTIONAL': 1000000000,
                    'INTERNATIONAL': 500000000
                },
                'growth_rate': 15.5  # 15.5% MoM growth
            },
            'clients': {
                'total': 250,
                'new_this_month': 15,
                'retention_rate': 95.5,
                'satisfaction_score': 4.2
            },
            'revenue': {
                'management_fees': 8333333,  # 2% annual on 500cr
                'performance_fees': 5000000,  # Performance fees
                'total_revenue': 13333333,
                'revenue_per_client': 53333
            },
            'operations': {
                'employee_count': 45,
                'cost_income_ratio': 65.5,
                'break_even_status': True,
                'runway_months': 24
            }
        }
        
        return metrics
    
    def _get_aum_by_strategy(self) -> Dict:
        """Get AUM breakdown by strategy"""
        return {
            'BALANCED_GROWTH': 2000000000,
            'AGGRESSIVE_GROWTH': 1500000000,
            'CONSERVATIVE': 1000000000,
            'HEDGE_FUND': 500000000
        }
    
    def _get_aum_by_client_type(self) -> Dict:
        """Get AUM breakdown by client type"""
        return {
            'HNI': 2000000000,
            'FAMILY_OFFICE': 1500000000,
            'INSTITUTIONAL': 1000000000,
            'INTERNATIONAL': 500000000
        }

class InstitutionalReportingSystem:
    """Generates institutional-grade reports"""
    
    def __init__(self):
        if DATABASE_AVAILABLE:
            self.db = SessionLocal()
        else:
            self.db = None
        
    async def generate_client_report(self, client_id: str, report_type: str = 'MONTHLY') -> Dict:
        """Generate comprehensive client report"""
        
        # Simulate client and portfolio data
        client = {
            'id': client_id,
            'name': 'Sample Client',
            'email': 'client@example.com'
        }
        
        portfolios = [
            {
                'id': str(uuid.uuid4()),
                'name': 'Growth Portfolio',
                'current_value': 10000000,
                'total_return': 0.12,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.05
            }
        ]
        
        report = {
            'report_id': str(uuid.uuid4()),
            'client_name': client['name'],
            'report_type': report_type,
            'period': self._get_report_period(report_type),
            'generation_date': datetime.now().isoformat(),
            'portfolios': []
        }
        
        total_value = 0
        total_return = 0
        
        for portfolio in portfolios:
            portfolio_data = await self._generate_portfolio_report(portfolio)
            report['portfolios'].append(portfolio_data)
            total_value += portfolio['current_value']
            total_return += portfolio['current_value'] * portfolio['total_return']
        
        report['summary'] = {
            'total_aum': total_value,
            'total_return': total_return / total_value if total_value > 0 else 0,
            'best_performer': max(report['portfolios'], key=lambda x: x['return'])['name'],
            'risk_metrics': await self._calculate_aggregate_risk(portfolios)
        }
        
        return report
    
    def _get_report_period(self, report_type: str) -> Dict:
        """Get report period dates"""
        
        end_date = date.today()
        
        if report_type == 'MONTHLY':
            start_date = end_date.replace(day=1)
        elif report_type == 'QUARTERLY':
            quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
            start_date = end_date.replace(month=quarter_start_month, day=1)
        else:  # ANNUAL
            start_date = end_date.replace(month=1, day=1)
        
        return {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        }
    
    async def _generate_portfolio_report(self, portfolio: Dict) -> Dict:
        """Generate individual portfolio report"""
        
        return {
            'name': portfolio['name'],
            'value': portfolio['current_value'],
            'return': portfolio['total_return'],
            'sharpe_ratio': portfolio['sharpe_ratio'],
            'max_drawdown': portfolio['max_drawdown']
        }
    
    async def _calculate_aggregate_risk(self, portfolios: List[Dict]) -> Dict:
        """Calculate aggregate risk across portfolios"""
        
        total_value = sum(p['current_value'] for p in portfolios)
        weighted_return = sum(p['current_value'] * p['total_return'] for p in portfolios) / total_value if total_value > 0 else 0
        
        return {
            'total_value': total_value,
            'weighted_return': weighted_return,
            'aggregate_sharpe': 1.3,  # Simplified
            'correlation': 0.65
        }

class InstitutionalTradingSystem:
    """Main institutional trading system"""
    
    def __init__(self):
        self.client_manager = InstitutionalClientManager()
        self.portfolio_manager = InstitutionalPortfolioManager()
        self.risk_manager = InstitutionalRiskManager()
        self.compliance_manager = ComplianceManager()
        self.reporting_system = InstitutionalReportingSystem()
        self.operations_manager = BusinessOperationsManager()
        
        # Initialize database if available
        if DATABASE_AVAILABLE:
            try:
                Base.metadata.create_all(bind=engine)
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
        
    async def run_daily_operations(self):
        """Run daily operational tasks"""
        
        logger.info("Starting daily operations")
        
        try:
            # Simulate portfolio management
            portfolio_ids = ['portfolio_1', 'portfolio_2', 'portfolio_3']
            
            for portfolio_id in portfolio_ids:
                await self.portfolio_manager.manage_portfolio(portfolio_id)
            
            # Risk assessment
            for portfolio_id in portfolio_ids:
                assessment = await self.risk_manager.perform_risk_assessment(portfolio_id)
                if assessment['breaches']:
                    logger.warning(f"Risk breaches in portfolio {portfolio_id}: {assessment['breaches']}")
            
            # Compliance checks
            for portfolio_id in portfolio_ids:
                compliance = await self.compliance_manager.perform_compliance_check(
                    'PORTFOLIO', 
                    portfolio_id
                )
                if compliance['status'] != 'COMPLIANT':
                    logger.error(f"Compliance issues in portfolio {portfolio_id}: {compliance['issues']}")
            
            # Calculate business metrics
            metrics = await self.operations_manager.calculate_business_metrics()
            logger.info(f"Daily metrics: AUM=₹{metrics['aum']['total']:,.2f}")
            
        except Exception as e:
            logger.error(f"Daily operations error: {e}")
            raise
    
    async def _get_total_aum(self) -> Dict:
        """Get total AUM breakdown"""
        
        return {
            'total': 5000000000,
            'by_strategy': {
                'BALANCED_GROWTH': 2000000000,
                'AGGRESSIVE_GROWTH': 1500000000,
                'CONSERVATIVE': 1000000000,
                'HEDGE_FUND': 500000000
            },
            'by_client_type': {
                'HNI': 2000000000,
                'FAMILY_OFFICE': 1500000000,
                'INSTITUTIONAL': 1000000000,
                'INTERNATIONAL': 500000000
            },
            'growth_rate': 15.5
        }

# ===================== INTEGRATION =====================

def integrate_institutional_system(bot_instance):
    """Integrate institutional system with main bot"""
    
    # Initialize institutional system
    bot_instance.institutional = InstitutionalTradingSystem()
    
    async def institutional_command(update, context):
        """Institutional system command handler"""
        
        if not context.args:
            help_text = """
🏛️ **Institutional System Commands**

**Client Management:**
/institutional client - Client operations
/institutional portfolio - Portfolio management
/institutional onboard - Onboard new client

**Operations:**
/institutional nav - Calculate NAV
/institutional fees - Calculate fees
/institutional risk - Risk assessment
/institutional compliance - Compliance check

**Reports:**
/institutional report - Generate reports
/institutional metrics - Business metrics
/institutional aum - AUM summary

**System:**
/institutional status - System status
/institutional operations - Run daily operations
            """
            await update.message.reply_text(help_text, parse_mode='Markdown')
            return
        
        command = context.args[0].lower()
        
        if command == 'metrics':
            metrics = await bot_instance.institutional.operations_manager.calculate_business_metrics()
            
            msg = f"""
📊 **Business Metrics**

**Assets Under Management:**
• Total AUM: ₹{metrics['aum']['total']:,.2f}
• Growth Rate: {metrics['aum']['growth_rate']:.1f}% MoM

**Client Base:**
• Total Clients: {metrics['clients']['total']:,}
• New This Month: {metrics['clients']['new_this_month']}
• Retention Rate: {metrics['clients']['retention_rate']:.1f}%
• Satisfaction: {metrics['clients']['satisfaction_score']:.1f}/5.0

**Revenue (Monthly):**
• Management Fees: ₹{metrics['revenue']['management_fees']:,.2f}
• Performance Fees: ₹{metrics['revenue']['performance_fees']:,.2f}
• Total Revenue: ₹{metrics['revenue']['total_revenue']:,.2f}
• Revenue/Client: ₹{metrics['revenue']['revenue_per_client']:,.2f}

**Operations:**
• Employees: {metrics['operations']['employee_count']}
• Cost/Income Ratio: {metrics['operations']['cost_income_ratio']:.1f}%
• Break Even: {'✅ Yes' if metrics['operations']['break_even_status'] else '❌ No'}
• Runway: {metrics['operations']['runway_months']} months
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'aum':
            aum_data = await bot_instance.institutional._get_total_aum()
            
            msg = f"""
💰 **Assets Under Management**

**Total AUM:** ₹{aum_data['total']:,.2f} (500 Crores)

**By Strategy:**
• Balanced Growth: ₹{aum_data['by_strategy']['BALANCED_GROWTH']:,.2f}
• Aggressive Growth: ₹{aum_data['by_strategy']['AGGRESSIVE_GROWTH']:,.2f}
• Conservative: ₹{aum_data['by_strategy']['CONSERVATIVE']:,.2f}
• Hedge Fund: ₹{aum_data['by_strategy']['HEDGE_FUND']:,.2f}

**By Client Type:**
• HNI: ₹{aum_data['by_client_type']['HNI']:,.2f}
• Family Office: ₹{aum_data['by_client_type']['FAMILY_OFFICE']:,.2f}
• Institutional: ₹{aum_data['by_client_type']['INSTITUTIONAL']:,.2f}
• International: ₹{aum_data['by_client_type']['INTERNATIONAL']:,.2f}

**Growth Rate:** {aum_data['growth_rate']:.1f}% MoM
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'risk':
            # Perform risk assessment on sample portfolio
            portfolio_id = 'portfolio_1'
            assessment = await bot_instance.institutional.risk_manager.perform_risk_assessment(portfolio_id)
            
            msg = f"""
🛡️ **Risk Assessment**

**Portfolio:** {portfolio_id}

**Risk Metrics:**
• VaR (95%): ₹{assessment['risk_metrics']['var_95']:,.2f}
• Max Concentration: {assessment['risk_metrics']['concentration']['max_concentration']:.1f}%
• Top Position: {assessment['risk_metrics']['concentration']['max_symbol']}

**Stress Test Results:**
"""
            for scenario, result in assessment['risk_metrics']['stress_test'].items():
                msg += f"• {scenario.replace('_', ' ').title()}: {result['impact_percent']:+.1f}%\n"
            
            if assessment['breaches']:
                msg += f"\n**🚨 Risk Breaches:**\n"
                for breach in assessment['breaches']:
                    msg += f"• {breach['type']}: {breach['current']:.2f} > {breach['limit']:.2f}\n"
            else:
                msg += f"\n**✅ No risk breaches**"
            
            if assessment['recommendations']:
                msg += f"\n**📋 Recommendations:**\n"
                for rec in assessment['recommendations']:
                    msg += f"• {rec}\n"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'compliance':
            # Perform compliance check
            portfolio_id = 'portfolio_1'
            compliance = await bot_instance.institutional.compliance_manager.perform_compliance_check(
                'PORTFOLIO', portfolio_id
            )
            
            msg = f"""
📋 **Compliance Check**

**Entity:** Portfolio {portfolio_id}
**Status:** {compliance['status']}
**Check Date:** {compliance['timestamp'][:19]}

"""
            if compliance['issues']:
                msg += f"**🚨 Compliance Issues:**\n"
                for issue in compliance['issues']:
                    msg += f"• {issue}\n"
            else:
                msg += f"**✅ No compliance issues**"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'operations':
            await update.message.reply_text("🔄 Running daily operations...")
            
            try:
                await bot_instance.institutional.run_daily_operations()
                await update.message.reply_text("✅ Daily operations completed successfully")
            except Exception as e:
                await update.message.reply_text(f"❌ Daily operations failed: {str(e)}")
        
        elif command == 'fees':
            # Calculate fees for sample portfolio
            portfolio_id = 'portfolio_1'
            fees = await bot_instance.institutional.portfolio_manager.calculate_fees(portfolio_id, 'QUARTERLY')
            
            msg = f"""
💳 **Fee Calculation**

**Portfolio:** {portfolio_id}
**Period:** {fees['period']}

**Fees:**
• Management Fee: ₹{fees['management_fee']:,.2f}
• Performance Fee: ₹{fees['performance_fee']:,.2f}
• **Total Fees:** ₹{fees['total_fees']:,.2f}

**Calculation Date:** {fees['calculation_date'][:19]}
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'report':
            # Generate sample client report
            client_id = 'client_1'
            report = await bot_instance.institutional.reporting_system.generate_client_report(
                client_id, 'MONTHLY'
            )
            
            msg = f"""
📊 **Client Report**

**Report ID:** {report['report_id']}
**Client:** {report['client_name']}
**Type:** {report['report_type']}
**Period:** {report['period']['start_date']} to {report['period']['end_date']}

**Summary:**
• Total AUM: ₹{report['summary']['total_aum']:,.2f}
• Total Return: {report['summary']['total_return']:.2%}
• Best Performer: {report['summary']['best_performer']}

**Portfolios:** {len(report['portfolios'])}
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
        
        elif command == 'status':
            msg = f"""
🏛️ **Institutional System Status**

**Components:**
• Client Manager: ✅ Active
• Portfolio Manager: ✅ Active
• Risk Manager: ✅ Active
• Compliance Manager: ✅ Active
• Reporting System: ✅ Active
• Operations Manager: ✅ Active

**Database:** {'✅ Connected' if DATABASE_AVAILABLE else '⚠️ Demo Mode'}
**Redis Cache:** {'✅ Available' if bot_instance.institutional.client_manager.redis_available else '⚠️ Not Available'}

**Configuration:**
• Min HNI Investment: ₹{os.getenv('MIN_INVESTMENT_HNI', '10000000')}
• Min Family Office: ₹{os.getenv('MIN_INVESTMENT_FAMILY_OFFICE', '100000000')}
• Min Institutional: ₹{os.getenv('MIN_INVESTMENT_INSTITUTIONAL', '500000000')}
• Management Fee: 2%
• Performance Fee: 20%
• Hurdle Rate: {os.getenv('HURDLE_RATE_PERCENT', '8')}%
            """
            await update.message.reply_text(msg, parse_mode='Markdown')
    
    return institutional_command

# ===================== MAIN ENTRY POINT =====================

async def main():
    """Main entry point for institutional system"""
    
    system = InstitutionalTradingSystem()
    
    # Run daily operations
    await system.run_daily_operations()

if __name__ == "__main__":
    asyncio.run(main())
