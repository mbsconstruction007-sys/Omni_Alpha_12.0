"""
STEP 12: ROLE-BASED ARCHITECTURE DEFINITIONS
Deep analysis of each component's role in the ecosystem
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional

# ============================================
# ROLE DEFINITIONS
# ============================================

class SystemRole(Enum):
    """Core roles in the ecosystem"""
    
    # Infrastructure Roles
    INFRASTRUCTURE_PROVIDER = "Provides core trading infrastructure to other institutions"
    PLATFORM_OPERATOR = "Operates the trading platform ecosystem"
    CLOUD_PROVIDER = "Offers cloud-based trading services"
    API_GATEWAY = "Manages and monetizes API access"
    
    # Market Roles
    MARKET_MAKER = "Provides liquidity across all markets"
    PRICE_DISCOVERER = "Determines fair market prices"
    LIQUIDITY_AGGREGATOR = "Aggregates liquidity from multiple sources"
    SYSTEMATIC_INTERNALIZER = "Matches orders internally"
    
    # Service Roles
    PRIME_BROKER = "Provides prime brokerage services"
    CLEARING_HOUSE = "Clears and settles trades"
    CUSTODIAN = "Safeguards client assets"
    ADMINISTRATOR = "Provides fund administration"
    
    # Product Roles
    PRODUCT_CREATOR = "Creates new financial products"
    INDEX_PROVIDER = "Creates and maintains indices"
    STRUCTURED_PRODUCTS_ISSUER = "Issues complex derivatives"
    TOKENIZATION_PLATFORM = "Tokenizes real-world assets"
    
    # Technology Roles
    AI_PROVIDER = "Provides AI/ML capabilities"
    DATA_VENDOR = "Sells data and analytics"
    REGTECH_PROVIDER = "Offers regulatory technology"
    RISK_SYSTEM_PROVIDER = "Provides risk management systems"
    
    # Regulatory Roles
    COMPLIANCE_MONITOR = "Monitors regulatory compliance"
    REPORTING_HUB = "Centralizes regulatory reporting"
    SURVEILLANCE_PROVIDER = "Provides market surveillance"
    AUDIT_FACILITATOR = "Facilitates regulatory audits"

@dataclass
class ComponentRole:
    """Detailed role definition for each component"""
    name: str
    primary_role: SystemRole
    secondary_roles: List[SystemRole]
    responsibilities: List[str]
    dependencies: List[str]
    revenue_streams: List[str]
    regulatory_requirements: List[str]
    performance_metrics: Dict[str, float]
    criticality: str  # "critical", "high", "medium", "low"

# ============================================
# COMPONENT ROLE MAPPINGS
# ============================================

ECOSYSTEM_COMPONENTS = {
    "TradingInfrastructure": ComponentRole(
        name="Trading Infrastructure as a Service",
        primary_role=SystemRole.INFRASTRUCTURE_PROVIDER,
        secondary_roles=[
            SystemRole.PLATFORM_OPERATOR,
            SystemRole.CLOUD_PROVIDER,
            SystemRole.API_GATEWAY
        ],
        responsibilities=[
            "Provide white-label trading platforms",
            "Offer API marketplace for trading services",
            "Maintain global colocation network",
            "Operate cloud trading infrastructure",
            "Ensure 99.999% uptime",
            "Scale to handle 1T+ daily transactions"
        ],
        dependencies=[
            "Data centers",
            "Network providers",
            "Cloud services",
            "Exchange connectivity"
        ],
        revenue_streams=[
            "Platform licensing fees",
            "API usage charges",
            "Infrastructure hosting fees",
            "Professional services"
        ],
        regulatory_requirements=[
            "Data protection (GDPR)",
            "Cloud service regulations",
            "Cross-border data transfer",
            "Technology export controls"
        ],
        performance_metrics={
            "uptime_percent": 99.999,
            "latency_ms": 0.1,
            "throughput_tps": 1000000,
            "clients": 1000
        },
        criticality="critical"
    ),
    
    "MarketMaking": ComponentRole(
        name="Global Market Making Operations",
        primary_role=SystemRole.MARKET_MAKER,
        secondary_roles=[
            SystemRole.PRICE_DISCOVERER,
            SystemRole.LIQUIDITY_AGGREGATOR,
            SystemRole.SYSTEMATIC_INTERNALIZER
        ],
        responsibilities=[
            "Provide continuous two-sided quotes",
            "Maintain orderly markets",
            "Reduce bid-ask spreads",
            "Absorb temporary imbalances",
            "Facilitate price discovery",
            "Internalize client order flow"
        ],
        dependencies=[
            "Exchange memberships",
            "Prime brokers",
            "Clearing firms",
            "Risk systems"
        ],
        revenue_streams=[
            "Spread capture",
            "Rebate optimization",
            "Payment for order flow",
            "Volatility trading"
        ],
        regulatory_requirements=[
            "Market maker obligations",
            "Best execution",
            "Market manipulation rules",
            "Capital requirements",
            "Reg NMS compliance"
        ],
        performance_metrics={
            "spread_bps": 1.0,
            "fill_rate_percent": 99.0,
            "inventory_turnover": 100,
            "daily_volume_usd": 100000000000
        },
        criticality="critical"
    ),
    
    "PrimeBrokerage": ComponentRole(
        name="Institutional Prime Brokerage",
        primary_role=SystemRole.PRIME_BROKER,
        secondary_roles=[
            SystemRole.CLEARING_HOUSE,
            SystemRole.CUSTODIAN,
            SystemRole.ADMINISTRATOR
        ],
        responsibilities=[
            "Provide leverage and financing",
            "Securities lending and borrowing",
            "Trade execution services",
            "Clearing and settlement",
            "Custody services",
            "Risk management",
            "Capital introduction"
        ],
        dependencies=[
            "Clearing houses",
            "Custodian banks",
            "Trading venues",
            "Credit providers"
        ],
        revenue_streams=[
            "Financing charges",
            "Securities lending fees",
            "Commission revenue",
            "Custody fees",
            "Performance fees"
        ],
        regulatory_requirements=[
            "Basel III capital rules",
            "Customer protection rules",
            "Segregation requirements",
            "Rehypothecation limits",
            "Margin rules"
        ],
        performance_metrics={
            "client_count": 500,
            "financing_book_usd": 50000000000,
            "securities_on_loan_usd": 100000000000,
            "revenue_per_client": 10000000
        },
        criticality="high"
    ),
    
    "AIIntelligence": ComponentRole(
        name="AI Superintelligence Platform",
        primary_role=SystemRole.AI_PROVIDER,
        secondary_roles=[
            SystemRole.DATA_VENDOR,
            SystemRole.RISK_SYSTEM_PROVIDER
        ],
        responsibilities=[
            "Develop AGI-level trading systems",
            "Operate swarm intelligence network",
            "Provide predictive analytics",
            "Offer AI-as-a-Service",
            "Continuous model improvement",
            "Quantum-classical hybrid computing"
        ],
        dependencies=[
            "GPU/TPU clusters",
            "Quantum processors",
            "Data providers",
            "Research teams"
        ],
        revenue_streams=[
            "AI model licensing",
            "Prediction API fees",
            "Custom model development",
            "Performance-based fees"
        ],
        regulatory_requirements=[
            "AI ethics guidelines",
            "Model explainability",
            "Data privacy",
            "Algorithmic accountability"
        ],
        performance_metrics={
            "model_accuracy_percent": 85,
            "prediction_latency_ms": 1,
            "models_deployed": 10000,
            "api_calls_daily": 1000000000
        },
        criticality="high"
    ),
    
    "DataEmpire": ComponentRole(
        name="Global Data Empire",
        primary_role=SystemRole.DATA_VENDOR,
        secondary_roles=[
            SystemRole.INDEX_PROVIDER,
            SystemRole.SURVEILLANCE_PROVIDER
        ],
        responsibilities=[
            "Collect all market data globally",
            "Process alternative data sources",
            "Create proprietary indices",
            "Provide market surveillance",
            "Offer data analytics services",
            "Monetize data assets"
        ],
        dependencies=[
            "Data sources",
            "Processing infrastructure",
            "Analytics teams",
            "Regulatory approvals"
        ],
        revenue_streams=[
            "Data subscription fees",
            "Index licensing",
            "Analytics services",
            "Surveillance fees"
        ],
        regulatory_requirements=[
            "Data licensing agreements",
            "Market data regulations",
            "Privacy compliance",
            "Cross-border data rules"
        ],
        performance_metrics={
            "data_sources": 10000,
            "daily_data_points": 1000000000000,
            "subscribers": 5000,
            "revenue_per_subscriber": 100000
        },
        criticality="high"
    ),
    
    "RegTechPlatform": ComponentRole(
        name="Regulatory Technology Platform",
        primary_role=SystemRole.REGTECH_PROVIDER,
        secondary_roles=[
            SystemRole.COMPLIANCE_MONITOR,
            SystemRole.REPORTING_HUB,
            SystemRole.AUDIT_FACILITATOR
        ],
        responsibilities=[
            "Automate regulatory compliance",
            "Provide real-time monitoring",
            "Centralize reporting",
            "Facilitate audits",
            "Offer compliance-as-a-Service",
            "Ensure regulatory adherence"
        ],
        dependencies=[
            "Regulatory databases",
            "Compliance teams",
            "Reporting systems",
            "Audit frameworks"
        ],
        revenue_streams=[
            "Compliance software fees",
            "Monitoring services",
            "Reporting fees",
            "Audit facilitation"
        ],
        regulatory_requirements=[
            "Regulatory approvals",
            "Data security standards",
            "Audit trail requirements",
            "Cross-jurisdiction compliance"
        ],
        performance_metrics={
            "regulations_covered": 1000,
            "clients_monitored": 2000,
            "compliance_rate": 99.9,
            "audit_success_rate": 100
        },
        criticality="high"
    ),
    
    "WealthManagement": ComponentRole(
        name="Global Wealth Management Platform",
        primary_role=SystemRole.ADMINISTRATOR,
        secondary_roles=[
            SystemRole.CUSTODIAN,
            SystemRole.PRODUCT_CREATOR
        ],
        responsibilities=[
            "Manage institutional wealth",
            "Provide custody services",
            "Create investment products",
            "Offer portfolio management",
            "Ensure regulatory compliance",
            "Maximize returns"
        ],
        dependencies=[
            "Investment teams",
            "Custody infrastructure",
            "Product development",
            "Risk management"
        ],
        revenue_streams=[
            "Management fees",
            "Performance fees",
            "Custody fees",
            "Product fees"
        ],
        regulatory_requirements=[
            "Investment advisor regulations",
            "Custody rules",
            "Fiduciary duties",
            "Client protection"
        ],
        performance_metrics={
            "aum_usd": 500000000000,
            "client_count": 1000,
            "performance_alpha": 0.05,
            "client_retention": 0.95
        },
        criticality="high"
    ),
    
    "LiquidityNetwork": ComponentRole(
        name="Global Liquidity Network",
        primary_role=SystemRole.LIQUIDITY_AGGREGATOR,
        secondary_roles=[
            SystemRole.SYSTEMATIC_INTERNALIZER,
            SystemRole.PRICE_DISCOVERER
        ],
        responsibilities=[
            "Aggregate global liquidity",
            "Optimize execution",
            "Reduce market impact",
            "Provide best execution",
            "Internalize flow",
            "Maximize fill rates"
        ],
        dependencies=[
            "Trading venues",
            "Market makers",
            "Execution algorithms",
            "Risk systems"
        ],
        revenue_streams=[
            "Execution fees",
            "Spread capture",
            "Rebate optimization",
            "Flow payments"
        ],
        regulatory_requirements=[
            "Best execution rules",
            "Market access regulations",
            "Systematic internalizer rules",
            "Transparency requirements"
        ],
        performance_metrics={
            "venues_connected": 200,
            "daily_volume_usd": 1000000000000,
            "fill_rate": 0.99,
            "price_improvement_bps": 2.5
        },
        criticality="critical"
    ),
    
    "SpaceInfrastructure": ComponentRole(
        name="Space-Based Trading Infrastructure",
        primary_role=SystemRole.INFRASTRUCTURE_PROVIDER,
        secondary_roles=[
            SystemRole.DATA_VENDOR,
            SystemRole.CLOUD_PROVIDER
        ],
        responsibilities=[
            "Deploy satellite network",
            "Provide global connectivity",
            "Enable space-based trading",
            "Offer low-latency communication",
            "Support space commerce",
            "Expand trading frontiers"
        ],
        dependencies=[
            "Satellite technology",
            "Launch capabilities",
            "Ground stations",
            "Regulatory approvals"
        ],
        revenue_streams=[
            "Connectivity fees",
            "Data services",
            "Trading infrastructure",
            "Space commerce"
        ],
        regulatory_requirements=[
            "Space regulations",
            "Satellite licensing",
            "Frequency allocation",
            "International treaties"
        ],
        performance_metrics={
            "satellites_deployed": 1000,
            "global_coverage": 1.0,
            "latency_ms": 10,
            "reliability": 0.999
        },
        criticality="medium"
    ),
    
    "ProductFactory": ComponentRole(
        name="Financial Product Factory",
        primary_role=SystemRole.PRODUCT_CREATOR,
        secondary_roles=[
            SystemRole.STRUCTURED_PRODUCTS_ISSUER,
            SystemRole.TOKENIZATION_PLATFORM
        ],
        responsibilities=[
            "Create innovative products",
            "Issue structured products",
            "Tokenize real assets",
            "Develop new instruments",
            "Customize solutions",
            "Drive financial innovation"
        ],
        dependencies=[
            "Product teams",
            "Legal expertise",
            "Risk management",
            "Market access"
        ],
        revenue_streams=[
            "Product fees",
            "Structuring fees",
            "Tokenization fees",
            "Performance fees"
        ],
        regulatory_requirements=[
            "Product regulations",
            "Securities laws",
            "Token regulations",
            "Innovation frameworks"
        ],
        performance_metrics={
            "products_created": 1000,
            "aum_products": 100000000000,
            "innovation_index": 0.9,
            "market_adoption": 0.8
        },
        criticality="medium"
    )
}

# ============================================
# ROLE ANALYSIS FUNCTIONS
# ============================================

def get_component_by_role(role: SystemRole) -> List[str]:
    """Get components that have a specific role"""
    components = []
    for name, component in ECOSYSTEM_COMPONENTS.items():
        if (component.primary_role == role or 
            role in component.secondary_roles):
            components.append(name)
    return components

def get_critical_components() -> List[str]:
    """Get all critical components"""
    return [name for name, component in ECOSYSTEM_COMPONENTS.items() 
            if component.criticality == "critical"]

def get_revenue_streams() -> Dict[str, List[str]]:
    """Get all revenue streams by component"""
    return {name: component.revenue_streams 
            for name, component in ECOSYSTEM_COMPONENTS.items()}

def calculate_total_revenue_potential() -> float:
    """Calculate total revenue potential across all components"""
    # This would use actual revenue projections
    # For now, return a mock calculation
    return 50000000000  # $50B annual potential

def get_ecosystem_dependencies() -> Dict[str, List[str]]:
    """Get dependency graph for all components"""
    return {name: component.dependencies 
            for name, component in ECOSYSTEM_COMPONENTS.items()}

def analyze_competitive_moats() -> Dict[str, List[str]]:
    """Analyze competitive moats for each component"""
    moats = {}
    
    for name, component in ECOSYSTEM_COMPONENTS.items():
        component_moats = []
        
        # Technology moats
        if "AI" in name or "Technology" in name:
            component_moats.extend(["technology_superiority", "patent_portfolio", "talent_concentration"])
        
        # Network effects
        if "Network" in name or "Platform" in name:
            component_moats.extend(["network_effects", "switching_costs", "ecosystem_lock_in"])
        
        # Data moats
        if "Data" in name:
            component_moats.extend(["data_monopoly", "proprietary_sources", "processing_advantage"])
        
        # Regulatory moats
        if "Reg" in name or "Compliance" in name:
            component_moats.extend(["regulatory_licenses", "compliance_expertise", "regulatory_relationships"])
        
        # Capital moats
        if "Prime" in name or "Wealth" in name:
            component_moats.extend(["capital_requirements", "balance_sheet_strength", "credit_rating"])
        
        moats[name] = component_moats
    
    return moats
