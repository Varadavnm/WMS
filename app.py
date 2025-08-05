import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

# LangChain imports
try:
    # Load .env with GOOGLE_API_KEY
    from dotenv import load_dotenv
    from langchain.agents import AgentType, initialize_agent
    from langchain.agents.agent_toolkits import SQLDatabaseToolkit
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.schema import AgentAction, AgentFinish
    from langchain.sql_database import SQLDatabase
    from langchain.tools import BaseTool, StructuredTool
    # Gemini model
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field
    load_dotenv()

    LANGCHAIN_AVAILABLE = True

except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    st.warning(
        "⚠️ LangChain or Gemini integration not installed. "
        "Run: `pip install langchain langchain-google-genai google-generativeai`"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Warehouse Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .ai-response {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .tool-execution {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Database setup
@st.cache_resource
def init_database():
    """Initialize the database with proper error handling"""
    try:
        conn = sqlite3.connect('wms.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sku_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                msku TEXT NOT NULL,
                product_name TEXT,
                marketplace TEXT DEFAULT 'default',
                category TEXT DEFAULT 'uncategorized',
                brand TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                msku TEXT,
                product_name TEXT NOT NULL,
                marketplace TEXT NOT NULL,
                quantity_sold INTEGER NOT NULL,
                unit_price REAL NOT NULL,
                total_amount REAL NOT NULL,
                sale_date TIMESTAMP NOT NULL,
                order_id TEXT,
                customer_info TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                msku TEXT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                stock_quantity INTEGER DEFAULT 0,
                marketplace TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                msku TEXT,
                title TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                date DATE NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create query_history table for AI queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_type TEXT DEFAULT 'ai',
                sql_generated TEXT,
                response_summary TEXT,
                success BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        return conn
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        st.error(f"Database initialization failed: {str(e)}")
        return None
class WMSQueryInput(BaseModel):
    query: str = Field(description="The user's natural language query about warehouse data")

class WMSSalesAnalysisInput(BaseModel):
    analysis_type: str = Field(description="Type of analysis: 'top_products', 'marketplace_performance', 'revenue_trends', 'inventory_status'")
    limit: Optional[int] = Field(default=10, description="Number of results to return")
    date_range: Optional[str] = Field(default="30", description="Date range in days")

class WMSMappingInput(BaseModel):
    action: str = Field(description="Action type: 'status', 'unmapped', 'create_mapping'")
    sku: Optional[str] = Field(default=None, description="SKU for mapping operations")
    msku: Optional[str] = Field(default=None, description="MSKU for mapping operations")

class LangChainWMSTools:
    def __init__(self, conn, google_api_key: str = None):
        self.conn = conn
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.db = None
        self.agent = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if LANGCHAIN_AVAILABLE and self.google_api_key:
            self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain components"""
        try:
            # Create SQLDatabase instance
            self.db = SQLDatabase.from_uri(f"sqlite:///wms.db")
            # This reads GOOGLE_API_KEY from .env
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro")

            
            # Create custom tools
            self.tools = [
                self._create_sales_analysis_tool(),
                self._create_mapping_tool(),
                self._create_inventory_tool(),
                self._create_chart_generator_tool()
            ]
            
            # Create SQL toolkit
            sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            sql_tools = sql_toolkit.get_tools()
            
            # Combine all tools
            all_tools = self.tools + sql_tools
            
            # Create agent
            self.agent = initialize_agent(
                all_tools,
                self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
            
            logger.info("LangChain initialized successfully")
            
        except Exception as e:
            logger.error(f"LangChain initialization failed: {str(e)}")
            self.agent = None
    
    def _create_sales_analysis_tool(self):
        """Create sales analysis tool"""
        def sales_analysis(analysis_type: str, limit: int = 10, date_range: str = "30") -> str:
            try:
                cursor = self.conn.cursor()
                
                if analysis_type == "top_products":
                    query = f'''
                        SELECT msku, product_name, 
                               SUM(total_amount) as revenue,
                               SUM(quantity_sold) as units_sold,
                               COUNT(*) as orders,
                               AVG(unit_price) as avg_price
                        FROM sales_records 
                        WHERE msku IS NOT NULL 
                        AND processed_at >= datetime('now', '-{date_range} days')
                        GROUP BY msku 
                        ORDER BY revenue DESC 
                        LIMIT {limit}
                    '''
                
                elif analysis_type == "marketplace_performance":
                    query = f'''
                        SELECT marketplace,
                               COUNT(*) as total_orders,
                               SUM(total_amount) as revenue,
                               AVG(total_amount) as avg_order_value,
                               SUM(quantity_sold) as units_sold
                        FROM sales_records 
                        WHERE processed_at >= datetime('now', '-{date_range} days')
                        GROUP BY marketplace
                        ORDER BY revenue DESC
                    '''
                
                elif analysis_type == "revenue_trends":
                    query = f'''
                        SELECT DATE(processed_at) as date,
                               SUM(total_amount) as daily_revenue,
                               COUNT(*) as daily_orders,
                               AVG(total_amount) as avg_order_value
                        FROM sales_records 
                        WHERE processed_at >= datetime('now', '-{date_range} days')
                        GROUP BY DATE(processed_at)
                        ORDER BY date DESC
                        LIMIT {limit}
                    '''
                
                else:
                    return f"Unknown analysis_type: {analysis_type}"
                
                cursor.execute(query)
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                if results:
                    df = pd.DataFrame(results, columns=columns)
                    return f"Analysis Results ({analysis_type}):\n{df.to_string(index=False)}"
                else:
                    return f"No data found for {analysis_type} analysis"
                    
            except Exception as e:
                return f"Error in sales analysis: {str(e)}"
        
        return StructuredTool.from_function(
            func=sales_analysis,
            name="sales_analysis",
            description="Analyze sales data with different analysis types: top_products, marketplace_performance, revenue_trends",
        )
    
    def _create_mapping_tool(self):
        """Create SKU mapping tool"""
        def mapping_operations(action: str, sku: str = None, msku: str = None) -> str:
            try:
                cursor = self.conn.cursor()
                
                if action == "status":
                    cursor.execute("SELECT COUNT(DISTINCT sku) FROM sales_records")
                    total_skus = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT sku) FROM sales_records WHERE msku IS NOT NULL")
                    mapped_skus = cursor.fetchone()[0]
                    
                    mapping_rate = (mapped_skus / total_skus * 100) if total_skus > 0 else 0
                    
                    return f"Mapping Status: {mapped_skus}/{total_skus} SKUs mapped ({mapping_rate:.1f}%)"
                
                elif action == "unmapped":
                    cursor.execute('''
                        SELECT sku, product_name, marketplace, COUNT(*) as occurrences
                        FROM sales_records 
                        WHERE msku IS NULL
                        GROUP BY sku
                        ORDER BY occurrences DESC
                        LIMIT 20
                    ''')
                    results = cursor.fetchall()
                    
                    if results:
                        df = pd.DataFrame(results, columns=['SKU', 'Product Name', 'Marketplace', 'Occurrences'])
                        return f"Top Unmapped SKUs:\n{df.to_string(index=False)}"
                    else:
                        return "No unmapped SKUs found"
                
                elif action == "create_mapping" and sku and msku:
                    cursor.execute('''
                        INSERT OR REPLACE INTO sku_mappings (sku, msku, product_name, marketplace)
                        VALUES (?, ?, ?, ?)
                    ''', (sku, msku, f"Product for {sku}", "default"))
                    self.conn.commit()
                    return f"Created mapping: {sku} -> {msku}"
                
                else:
                    return f"Invalid mapping action or missing parameters"
                    
            except Exception as e:
                return f"Error in mapping operations: {str(e)}"
        
        return StructuredTool.from_function(
            func=mapping_operations,
            name="mapping_operations",
            description="Handle SKU mapping operations: status, unmapped, create_mapping",
        )
    
    def _create_inventory_tool(self):
        """Create inventory management tool"""
        def inventory_operations(operation: str, sku: str = None) -> str:
            try:
                cursor = self.conn.cursor()
                
                if operation == "stock_levels":
                    cursor.execute('''
                        SELECT p.sku, p.msku, p.name, p.stock_quantity, p.price,
                               COALESCE(SUM(sr.quantity_sold), 0) as sold_last_30_days
                        FROM products p
                        LEFT JOIN sales_records sr ON p.sku = sr.sku 
                        AND sr.processed_at >= datetime('now', '-30 days')
                        GROUP BY p.sku
                        ORDER BY p.stock_quantity ASC
                        LIMIT 20
                    ''')
                    results = cursor.fetchall()
                    
                    if results:
                        columns = ['SKU', 'MSKU', 'Product Name', 'Stock', 'Price', 'Sold (30d)']
                        df = pd.DataFrame(results, columns=columns)
                        return f"Current Stock Levels:\n{df.to_string(index=False)}"
                    else:
                        return "No inventory data found"
                
                elif operation == "low_stock":
                    cursor.execute('''
                        SELECT sku, msku, name, stock_quantity, price
                        FROM products 
                        WHERE stock_quantity < 10
                        ORDER BY stock_quantity ASC
                    ''')
                    results = cursor.fetchall()
                    
                    if results:
                        columns = ['SKU', 'MSKU', 'Product Name', 'Stock', 'Price']
                        df = pd.DataFrame(results, columns=columns)
                        return f"Low Stock Items:\n{df.to_string(index=False)}"
                    else:
                        return "No low stock items found"
                
                elif operation == "stock_for_sku" and sku:
                    cursor.execute('''
                        SELECT sku, msku, name, stock_quantity, price
                        FROM products 
                        WHERE sku = ? OR msku = ?
                    ''', (sku, sku))
                    result = cursor.fetchone()
                    
                    if result:
                        return f"Stock for {sku}: {result[3]} units (Price: ₹{result[4]})"
                    else:
                        return f"No inventory record found for {sku}"
                
                else:
                    return "Invalid inventory operation"
                    
            except Exception as e:
                return f"Error in inventory operations: {str(e)}"
        
        return StructuredTool.from_function(
            func=inventory_operations,
            name="inventory_operations",
            description="Handle inventory operations: stock_levels, low_stock, stock_for_sku",
        )
    
    def _create_chart_generator_tool(self):
        """Create chart generation recommendations"""
        def chart_recommendations(query_type: str, data_summary: str) -> str:
            recommendations = {
                "top_products": "Bar chart showing product names vs revenue with color coding",
                "marketplace_performance": "Pie chart for revenue distribution, bar chart for order counts",
                "revenue_trends": "Line chart with markers showing daily revenue over time",
                "stock_levels": "Horizontal bar chart showing current stock levels",
                "mapping_status": "Donut chart showing mapped vs unmapped SKUs percentage"
            }
            
            return f"Chart Recommendation for {query_type}: {recommendations.get(query_type, 'Table display recommended')}"
        
        return StructuredTool.from_function(
            func=chart_recommendations,
            name="chart_recommendations",
            description="Get chart type recommendations for different query types",
        )
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query using LangChain agent"""
        if not self.agent:
            return {
                "success": False,
                "error": "LangChain agent not initialized. Check google API key.",
                "fallback": True
            }
        
        try:
            # Save query to history
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO query_history (query_text, query_type)
                VALUES (?, ?)
            ''', (user_query, 'langchain'))
            self.conn.commit()
            
            # Process with agent
            response = self.agent.run(user_query)
            
            # Update query history with response
            cursor.execute('''
                UPDATE query_history 
                SET response_summary = ?, success = TRUE
                WHERE id = (SELECT MAX(id) FROM query_history)
            ''', (response[:500],))
            self.conn.commit()
            
            return {
                "success": True,
                "response": response,
                "query": user_query,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"LangChain query processing failed: {str(e)}")
            
            # Update query history with error
            cursor.execute('''
                UPDATE query_history 
                SET response_summary = ?, success = FALSE
                WHERE id = (SELECT MAX(id) FROM query_history)
            ''', (f"Error: {str(e)}",))
            self.conn.commit()
            
            return {
                "success": False,
                "error": str(e),
                "query": user_query,
                "fallback": True
            }

# SKU Mapper Class (unchanged)
class SKUMapper:
    def __init__(self, conn):
        self.conn = conn
        self.marketplace_formats = {
            'amazon': {'sku_column': 'ASIN', 'name_column': 'Product Name', 'price_column': 'Price'},
            'flipkart': {'sku_column': 'FSN', 'name_column': 'Product Title', 'price_column': 'Selling Price'},
            'meesho': {'sku_column': 'Product ID', 'name_column': 'Product Name', 'price_column': 'Price'},
            'myntra': {'sku_column': 'Style ID', 'name_column': 'Product Name', 'price_column': 'MRP'},
            'default': {'sku_column': 'SKU', 'name_column': 'Product Name', 'price_column': 'Price'}
        }
    
    def validate_sku_format(self, sku: str) -> bool:
        """Validate SKU format"""
        if not sku or len(sku.strip()) == 0:
            return False
        return bool(re.match(r'^[A-Za-z0-9_\-\.]+$', sku.strip()))
    
    def get_msku_from_db(self, sku: str) -> Optional[str]:
        """Get MSKU from database for given SKU"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT msku FROM sku_mappings WHERE sku = ?", (sku,))
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting MSKU: {str(e)}")
            return None
    
    def add_mapping_to_db(self, sku: str, msku: str, product_name: str = "", marketplace: str = "default"):
        """Add or update SKU mapping in database"""
        if not self.validate_sku_format(sku):
            raise ValueError(f"Invalid SKU format: {sku}")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sku_mappings (sku, msku, product_name, marketplace)
                VALUES (?, ?, ?, ?)
            ''', (sku, msku, product_name, marketplace))
            self.conn.commit()
            logger.info(f"Added mapping: {sku} -> {msku}")
        except Exception as e:
            logger.error(f"Error adding mapping: {str(e)}")
            raise
    
    def get_all_mappings(self) -> pd.DataFrame:
        """Get all SKU mappings as DataFrame"""
        try:
            return pd.read_sql_query("SELECT * FROM sku_mappings ORDER BY created_at DESC", self.conn)
        except Exception as e:
            logger.error(f"Error getting mappings: {str(e)}")
            return pd.DataFrame()

# Airtable Integration (unchanged)
class AirtableIntegration:
    def __init__(self, api_key: str, base_id: str):
        self.api_key = api_key
        self.base_id = base_id
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        } if api_key else {}
    
    def create_record(self, table_name: str, fields: Dict[str, Any]) -> bool:
        """Create a record in Airtable"""
        if not self.api_key or not self.base_id:
            logger.warning("Airtable not configured")
            return False
            
        url = f'https://api.airtable.com/v0/{self.base_id}/{table_name}'
        data = {'fields': fields}
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully created Airtable record in {table_name}")
            return True
        except requests.exceptions.RequestException as e:
            error_msg = f"Airtable sync failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    error_msg = "Airtable authentication failed. Check your API key."
                elif e.response.status_code == 404:
                    error_msg = f"Table '{table_name}' not found. Check your Base ID and table name."
                elif e.response.status_code == 422:
                    error_msg = "Invalid data format for Airtable. Check field names and types."
            
            logger.error(error_msg)
            return False
    
    def test_connection(self) -> bool:
        """Test if the connection to Airtable is working"""
        if not self.api_key or not self.base_id:
            return False
        
        url = f'https://api.airtable.com/v0/{self.base_id}/Test'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code in [200, 404]  # 404 means table doesn't exist but connection works
        except:
            return False

# Analytics Engine (unchanged)
class AnalyticsEngine:
    def __init__(self, conn):
        self.conn = conn
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics"""
        try:
            cursor = self.conn.cursor()
            metrics = {}
            
            # Basic metrics
            cursor.execute("SELECT COUNT(*) FROM sales_records")
            metrics['total_records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sku_mappings")
            metrics['total_mappings'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(total_amount) FROM sales_records")
            result = cursor.fetchone()
            metrics['total_revenue'] = result[0] if result and result[0] else 0
            
            cursor.execute("SELECT COUNT(DISTINCT msku) FROM sales_records WHERE msku IS NOT NULL")
            metrics['unique_products'] = cursor.fetchone()[0]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {str(e)}")
            return {
                'total_records': 0, 'total_mappings': 0, 'total_revenue': 0, 'unique_products': 0
            }

# Utility functions (unchanged)
def find_column(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """Find column by name (case-insensitive)"""
    for col in df.columns:
        if target_col.lower() in col.lower():
            return col
    return None

def load_sample_data(sku_mapper):
    """Load sample data if database is empty"""
    try:
        cursor = sku_mapper.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sku_mappings")
        count = cursor.fetchone()[0]
        
        if count == 0:
            sample_mappings = [
                ("Chimmy Pillow", "CSTE_0001_ST_Bts_Pillow_Chimmy", "Chimmy Pillow"),
                ("Tata Pillow", "CSTE_0002_ST_Bts_Pillow_Tata", "Tata Pillow"),
                ("Koya Pillow", "CSTE_0003_ST_Bts_Pillow_Koya", "Koya Pillow"),
                ("Mang Pillow", "CSTE_0004_ST_Bts_Pillow_Mang", "Mang Pillow"),
                ("Shooky Pillow", "CSTE_0005_ST_Bts_Pillow_Shooky", "Shooky Pillow"),
                ("Van Pillow", "CSTE_0006_ST_Bts_Pillow_Van", "Van Pillow"),
                ("Cooky Pillow", "CSTE_0007_ST_Bts_Pillow_Cooky", "Cooky Pillow"),
                ("Rj Pillow", "CSTE_0008_ST_Bts_Pillow_Rj", "Rj Pillow"),
                ("HBD_Red_MB", "CSTE_0013_MB_HBD_Red", "HBD Red MB"),
                ("HBD_Brown_MB", "CSTE_0014_MB_HBD_Brown", "HBD Brown MB"),
                ("HBD_Black_MB", "CSTE_0015_MB_HBD_Black", "HBD Black MB"),
            ]
            
            # Add sample sales data
            sample_sales = []
            for i, (sku, msku, product_name) in enumerate(sample_mappings):
                sample_sales.extend([
                    (sku, msku, product_name, 'amazon', 2, 299.99, 599.98, datetime.now() - timedelta(days=i)),
                    (sku, msku, product_name, 'flipkart', 1, 249.99, 249.99, datetime.now() - timedelta(days=i+1)),
                ])
            
            for sku, msku, product_name in sample_mappings:
                try:
                    sku_mapper.add_mapping_to_db(sku, msku, product_name)
                except Exception as e:
                    logger.warning(f"Could not add sample mapping: {e}")
            
            # Add sample sales records
            cursor = sku_mapper.conn.cursor()
            for sale_data in sample_sales:
                try:
                    cursor.execute('''
                        INSERT INTO sales_records 
                        (sku, msku, product_name, marketplace, quantity_sold, unit_price, total_amount, sale_date, order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (*sale_data, f"ORDER_{hash(str(sale_data))}"))
                except Exception as e:
                    logger.warning(f"Could not add sample sales: {e}")
            
            sku_mapper.conn.commit()
            
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")

# Initialize components
conn = init_database()
if conn is None:
    st.error("Failed to initialize database. Please check your setup.")
    st.stop()

sku_mapper = SKUMapper(conn)
analytics = AnalyticsEngine(conn)

# Initialize LangChain tools
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_tools = LangChainWMSTools(conn, openai_api_key) if LANGCHAIN_AVAILABLE else None

# Main header
st.markdown("""
<div class="main-header">
    <h1>📦 Warehouse Management System</h1>
    <p>Complete solution for SKU mapping, sales data processing, and AI-powered analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Load Airtable config
    airtable_api_key = os.getenv('AIRTABLE_API_KEY')
    airtable_base_id = st.text_input(
        "Airtable Base ID", 
        help="Enter your Airtable base ID",
        key="sidebar_airtable_base_id"
    )
    
    # OpenAI API Key configuration
    st.subheader("🤖 AI Configuration")
    if openai_api_key:
        st.success("✅ OpenAI API Key loaded from environment")
        if langchain_tools and langchain_tools.agent:
            st.success("✅ LangChain agent initialized")
        else:
            st.warning("⚠️ LangChain agent not initialized")
    else:
        st.error("❌ OPENAI_API_KEY not found in .env file")
        manual_openai_key = st.text_input(
            "Manual OpenAI API Key", 
            type="password", 
            help="Override environment OpenAI API key",
            key="sidebar_manual_openai_key"
        )
        if manual_openai_key:
            openai_api_key = manual_openai_key
            # Reinitialize LangChain tools with new key
            if LANGCHAIN_AVAILABLE:
                langchain_tools = LangChainWMSTools(conn, openai_api_key)
            st.warning("Using manual OpenAI API key override")
    
    # Show connection status
    if airtable_api_key:
        st.success("✅ Airtable API Key loaded from environment")
    else:
        st.error("❌ AIRTABLE_API_KEY not found in .env file")
    
    # Add manual override option for testing
    with st.expander("Override Airtable API Key (for testing)"):
        manual_api_key = st.text_input(
            "Manual Airtable API Key", 
            type="password", 
            help="Override environment API key",
            key="sidebar_manual_api_key"
        )
        if manual_api_key:
            airtable_api_key = manual_api_key
            st.warning("Using manual Airtable API key override")
    
    if st.button("Load Sample Data"):
        load_sample_data(sku_mapper)
        st.success("Sample data loaded!")
        st.rerun()

# Initialize Airtable integration
airtable = AirtableIntegration(airtable_api_key, airtable_base_id) if airtable_api_key and airtable_base_id else None

# Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📤 Upload Data", "🔄 SKU Mapping", "📈 Analytics", "🤖 AI Query"])

# Tab 1: Dashboard
with tab1:
    st.header("📊 Dashboard Overview")
    
    # Get dashboard metrics
    metrics = analytics.get_dashboard_metrics()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['total_records']:,}</h3>
            <p>Total Sales Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>₹{metrics['total_revenue']:,.2f}</h3>
            <p>Total Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['total_mappings']:,}</h3>
            <p>SKU Mappings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['unique_products']:,}</h3>
            <p>Unique Products</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by marketplace
        try:
            marketplace_data = pd.read_sql_query('''
                SELECT marketplace, COUNT(*) as sales_count, SUM(total_amount) as revenue
                FROM sales_records 
                GROUP BY marketplace
            ''', conn)
            
            if not marketplace_data.empty:
                fig = px.pie(
                    marketplace_data, 
                    values='revenue', 
                    names='marketplace',
                    title='Revenue by Marketplace',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No marketplace data available")
        except Exception as e:
            st.error(f"Error loading marketplace data: {str(e)}")
    
    with col2:
        # Top products
        try:
            top_products = pd.read_sql_query('''
                SELECT msku, product_name, SUM(quantity_sold) as total_sold, SUM(total_amount) as revenue
                FROM sales_records 
                WHERE msku IS NOT NULL
                GROUP BY msku 
                ORDER BY revenue DESC 
                LIMIT 10
            ''', conn)
            
            if not top_products.empty:
                fig = px.bar(
                    top_products.head(5), 
                    x='revenue', 
                    y='product_name',
                    title='Top 5 Products by Revenue',
                    orientation='h',
                    color='revenue',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No product data available")
        except Exception as e:
            st.error(f"Error loading product data: {str(e)}")
    
    # Recent sales table
    st.subheader("Recent Sales Records")
    try:
        recent_sales = pd.read_sql_query('''
            SELECT sku, msku, product_name, marketplace, quantity_sold, unit_price, total_amount, processed_at
            FROM sales_records 
            ORDER BY processed_at DESC 
            LIMIT 10
        ''', conn)
        
        if not recent_sales.empty:
            st.dataframe(recent_sales, use_container_width=True)
        else:
            st.info("No sales records found. Upload some data to get started!")
    except Exception as e:
        st.error(f"Error loading recent sales: {str(e)}")

# Tab 2: Upload Data
with tab2:
    st.header("📤 Upload Sales Data")
    
    # Marketplace selection
    marketplace = st.selectbox(
        "Select Marketplace",
        options=list(sku_mapper.marketplace_formats.keys()),
        help="Choose the marketplace format for your data"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="Upload your sales data file (CSV or Excel)"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} records.")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column mapping
            st.subheader("Column Mapping")
            fmt = sku_mapper.marketplace_formats[marketplace]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sku_col_found = find_column(df, fmt['sku_column'])
                sku_column = st.selectbox(
                    "SKU Column",
                    options=df.columns.tolist(),
                    index=0 if sku_col_found is None else df.columns.tolist().index(sku_col_found)
                )
            
            with col2:
                name_col_found = find_column(df, fmt['name_column'])
                name_column = st.selectbox(
                    "Product Name Column",
                    options=['None'] + df.columns.tolist(),
                    index=0 if name_col_found is None else df.columns.tolist().index(name_col_found) + 1
                )
            
            with col3:
                price_col_found = find_column(df, fmt['price_column'])
                price_column = st.selectbox(
                    "Price Column",
                    options=['None'] + df.columns.tolist(),
                    index=0 if price_col_found is None else df.columns.tolist().index(price_col_found) + 1
                )
            
            # Process data button
            if st.button("Process Sales Data", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {
                    'total_records': len(df),
                    'processed': 0,
                    'mapped': 0,
                    'not_mapped': 0,
                    'errors': []
                }
                
                cursor = conn.cursor()
                
                for idx, row in df.iterrows():
                    try:
                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing record {idx + 1} of {len(df)}")
                        
                        # Extract data
                        sku = str(row[sku_column]).strip() if pd.notna(row[sku_column]) else f"UNKNOWN_{idx}"
                        product_name = str(row[name_column]).strip() if name_column != 'None' and pd.notna(row[name_column]) else "Unknown Product"
                        
                        try:
                            price = float(row[price_column]) if price_column != 'None' and pd.notna(row[price_column]) else 0.0
                        except (ValueError, TypeError):
                            price = 0.0
                        
                        if sku and sku != 'nan':
                            # Get or create MSKU
                            msku = sku_mapper.get_msku_from_db(sku)
                            if not msku:
                                # Auto-generate MSKU
                                timestamp = datetime.now().strftime("%Y%m%d")
                                msku = f"CSTE_{timestamp}_{marketplace.upper()}_{sku[:10]}"
                                sku_mapper.add_mapping_to_db(sku, msku, product_name, marketplace)
                                results['not_mapped'] += 1
                            else:
                                results['mapped'] += 1
                            
                            # Insert sales record
                            cursor.execute('''
                                INSERT INTO sales_records 
                                (sku, msku, product_name, marketplace, quantity_sold, unit_price, total_amount, sale_date, order_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (sku, msku, product_name, marketplace, 1, price, price, datetime.now(), f"ORDER_{idx}"))
                            
                            # Sync to Airtable if configured
                            if airtable:
                                airtable_fields = {
                                    'SKU': sku,
                                    'MSKU': msku,
                                    'Product Name': product_name,
                                    'Marketplace': marketplace,
                                    'Price': price,
                                    'Sale Date': datetime.now().isoformat()
                                }
                                airtable.create_record('Sales Data', airtable_fields)
                            
                            results['processed'] += 1
                    
                    except Exception as e:
                        results['errors'].append(f"Row {idx}: {str(e)}")
                
                conn.commit()
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Show results
                st.success(f"""
                ✅ Processing Complete!
                - Total Records: {results['total_records']}
                - Successfully Processed: {results['processed']}
                - Already Mapped: {results['mapped']}
                - New Mappings Created: {results['not_mapped']}
                - Errors: {len(results['errors'])}
                """)
                
                if results['errors']:
                    with st.expander("View Errors"):
                        for error in results['errors'][:10]:  # Show first 10 errors
                            st.error(error)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 3: SKU Mapping
with tab3:
    st.header("🔄 SKU to MSKU Mapping")
    
    # Add new mapping
    st.subheader("Add New Mapping")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_sku = st.text_input("SKU", key="new_mapping_sku")
    
    with col2:
        new_msku = st.text_input("MSKU", key="new_mapping_msku")
    
    with col3:
        new_product_name = st.text_input("Product Name", key="new_mapping_product_name")
    
    new_marketplace = st.selectbox(
        "Marketplace",
        options=list(sku_mapper.marketplace_formats.keys()),
        key="new_mapping_marketplace"
    )
    
    if st.button("Add Mapping", type="primary"):
        if new_sku and new_msku:
            try:
                sku_mapper.add_mapping_to_db(new_sku, new_msku, new_product_name, new_marketplace)
                st.success(f"Mapping added: {new_sku} → {new_msku}")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding mapping: {str(e)}")
        else:
            st.error("Both SKU and MSKU are required")
    
    st.divider()
    
    # View existing mappings
    st.subheader("Existing Mappings")
    
    # Search functionality
    search_term = st.text_input(
        "Search mappings", 
        placeholder="Enter SKU, MSKU, or product name...",
        key="mapping_search_term"
    )
    
    # Get mappings
    try:
        if search_term:
            mappings_df = pd.read_sql_query('''
                SELECT * FROM sku_mappings 
                WHERE sku LIKE ? OR msku LIKE ? OR product_name LIKE ?
                ORDER BY created_at DESC
            ''', conn, params=[f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'])
        else:
            mappings_df = pd.read_sql_query('SELECT * FROM sku_mappings ORDER BY created_at DESC LIMIT 100', conn)
        
        if not mappings_df.empty:
            # Display mappings
            st.dataframe(mappings_df, use_container_width=True)
        else:
            st.info("No mappings found. Add some mappings or upload data to get started!")
    except Exception as e:
        st.error(f"Error loading mappings: {str(e)}")
    
    # Bulk upload mappings
    st.subheader("Bulk Upload Mappings")
    mapping_file = st.file_uploader(
        "Upload mapping file (CSV with columns: sku, msku, product_name, marketplace)",
        type=['csv']
    )
    
    if mapping_file is not None:
        try:
            mapping_df = pd.read_csv(mapping_file)
            st.write("Preview:")
            st.dataframe(mapping_df.head())
            
            if st.button("Import Mappings"):
                cursor = conn.cursor()
                imported = 0
                for _, row in mapping_df.iterrows():
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO sku_mappings (sku, msku, product_name, marketplace)
                            VALUES (?, ?, ?, ?)
                        ''', (row.get('sku', ''), row.get('msku', ''), 
                              row.get('product_name', ''), row.get('marketplace', 'default')))
                        imported += 1
                    except Exception as e:
                        st.error(f"Error importing row: {e}")
                
                conn.commit()
                st.success(f"Imported {imported} mappings!")
                st.rerun()
        
        except Exception as e:
            st.error(f"Error reading mapping file: {str(e)}")

# Tab 4: Analytics
with tab4:
    st.header("📈 Analytics Dashboard")
    
    # Time range selector
    col1, col2 = st.columns([1, 3])
    with col1:
        date_range = st.selectbox(
            "Time Range",
            ["7 days", "30 days", "90 days", "All time"],
            index=1
        )
    
    days_map = {"7 days": 7, "30 days": 30, "90 days": 90, "All time": 365*10}
    days = days_map[date_range]
    
    # Revenue trends
    st.subheader("📈 Revenue Trends")
    try:
        revenue_data = pd.read_sql_query(f'''
            SELECT DATE(processed_at) as date,
                   SUM(total_amount) as revenue,
                   COUNT(*) as orders,
                   AVG(total_amount) as avg_order_value
            FROM sales_records 
            WHERE processed_at >= datetime('now', '-{days} days')
            GROUP BY DATE(processed_at)
            ORDER BY date
        ''', conn)
        
        if not revenue_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue trend line chart
                fig = px.line(
                    revenue_data, 
                    x='date', 
                    y='revenue',
                    title=f'Daily Revenue Trend ({date_range})',
                    markers=True
                )
                fig.update_traces(line_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Orders trend
                fig = px.bar(
                    revenue_data, 
                    x='date', 
                    y='orders',
                    title=f'Daily Orders ({date_range})',
                    color='orders',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue data available for the selected time range")
    except Exception as e:
        st.error(f"Error loading revenue trends: {str(e)}")
    
    st.divider()
    
    # Product performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Products")
        try:
            top_products = pd.read_sql_query(f'''
                SELECT msku, product_name, 
                       SUM(total_amount) as revenue,
                       SUM(quantity_sold) as units_sold,
                       COUNT(*) as orders,
                       AVG(unit_price) as avg_price
                FROM sales_records 
                WHERE msku IS NOT NULL 
                AND processed_at >= datetime('now', '-{days} days')
                GROUP BY msku 
                ORDER BY revenue DESC 
                LIMIT 10
            ''', conn)
            
            if not top_products.empty:
                # Display table
                st.dataframe(
                    top_products[['product_name', 'revenue', 'units_sold', 'orders']].round(2),
                    use_container_width=True
                )
                
                # Chart
                fig = px.bar(
                    top_products.head(5), 
                    x='product_name', 
                    y='revenue',
                    title='Top 5 Products by Revenue',
                    color='revenue',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No product data available")
        except Exception as e:
            st.error(f"Error loading product performance: {str(e)}")
    
    with col2:
        st.subheader("🛒 Marketplace Performance")
        try:
            marketplace_perf = pd.read_sql_query(f'''
                SELECT marketplace,
                       COUNT(*) as total_orders,
                       SUM(total_amount) as revenue,
                       AVG(total_amount) as avg_order_value,
                       SUM(quantity_sold) as units_sold
                FROM sales_records 
                WHERE processed_at >= datetime('now', '-{days} days')
                GROUP BY marketplace
                ORDER BY revenue DESC
            ''', conn)
            
            if not marketplace_perf.empty:
                # Display table
                st.dataframe(marketplace_perf.round(2), use_container_width=True)
                
                # Pie chart for revenue distribution
                fig = px.pie(
                    marketplace_perf, 
                    values='revenue', 
                    names='marketplace',
                    title='Revenue Distribution by Marketplace'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No marketplace data available")
        except Exception as e:
            st.error(f"Error loading marketplace performance: {str(e)}")
    
    st.divider()
    
    # SKU Mapping analysis
    st.subheader("🔍 SKU Mapping Analysis")
    try:
        # Get mapping statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT sku) FROM sales_records")
        total_skus = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT sku) FROM sales_records WHERE msku IS NOT NULL")
        mapped_skus = cursor.fetchone()[0]
        
        unmapped_skus = total_skus - mapped_skus
        mapping_rate = (mapped_skus / total_skus * 100) if total_skus > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total SKUs", total_skus)
        with col2:
            st.metric("Mapped SKUs", mapped_skus)
        with col3:
            st.metric("Mapping Rate", f"{mapping_rate:.1f}%")
        
        # Mapping status chart
        mapping_data = pd.DataFrame({
            'Status': ['Mapped', 'Unmapped'],
            'Count': [mapped_skus, unmapped_skus]
        })
        
        if total_skus > 0:
            fig = px.pie(
                mapping_data, 
                values='Count', 
                names='Status',
                title='SKU Mapping Status',
                color_discrete_map={'Mapped': '#28a745', 'Unmapped': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show unmapped SKUs
        if unmapped_skus > 0:
            st.subheader("❗ Unmapped SKUs")
            unmapped_data = pd.read_sql_query('''
                SELECT sku, product_name, marketplace, COUNT(*) as occurrences,
                       SUM(total_amount) as lost_revenue
                FROM sales_records 
                WHERE msku IS NULL
                GROUP BY sku
                ORDER BY lost_revenue DESC
                LIMIT 20
            ''', conn)
            
            if not unmapped_data.empty:
                st.dataframe(unmapped_data, use_container_width=True)
                st.warning(f"💡 Consider mapping these SKUs to improve data accuracy. Total potential lost tracking: ₹{unmapped_data['lost_revenue'].sum():.2f}")
    
    except Exception as e:
        st.error(f"Error loading mapping analysis: {str(e)}")

# Tab 5: Enhanced AI Query with LangChain
with tab5:
    st.header("🤖 AI-Powered Data Query")

    if not LANGCHAIN_AVAILABLE:
        st.error("⚠️ LangChain not available. Please install with: `pip install langchain openai`")
        st.info("Falling back to basic query functionality...")
    elif not openai_api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or use the sidebar.")
    elif not langchain_tools or not langchain_tools.agent:
        st.error("❌ LangChain agent not initialized. Check your OpenAI API key and configuration.")
    else:
        st.success("✅ AI Assistant ready! Ask me anything about your warehouse data.")
    
    # Natural Language Query Input
    st.subheader("💬 Natural Language Query")
    
    with st.expander("💡 Example Queries", expanded=False):
        st.markdown("""
        **Sales Analysis:**
        - "Show me the top 10 products by revenue in the last 30 days"
        - "Which marketplace is performing best this month?"
        - "What's the revenue trend for the last week?"
        
        **Inventory & Mapping:**
        - "How many SKUs are not mapped to MSKUs?"
        - "Show me the unmapped SKUs with highest revenue"
        - "What's our current mapping status?"
        
        **Advanced Questions:**
        - "Which products have declining sales this month?"
        - "Compare marketplace performance across different time periods"
        - "Find products with low stock levels"
        """)

    user_query = st.text_area(
        "Enter your query:",
        placeholder="e.g., 'Show top 5 products by revenue this month'",
        key="ai_query_input",
        height=100
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        execute_query = st.button("🚀 Ask AI", type="primary", disabled=not user_query)
    with col2:
        use_langchain = st.checkbox(
            "Use Advanced AI (LangChain)", 
            value=True, 
            disabled=not (LANGCHAIN_AVAILABLE and openai_api_key and langchain_tools and langchain_tools.agent),
            help="Uses OpenAI + LangChain for intelligent query processing"
        )

    if execute_query and user_query:
        with st.spinner("🤔 AI is thinking..."):
            if use_langchain and langchain_tools and langchain_tools.agent:
                try:
                    result = langchain_tools.process_query(user_query)
                    if result['success']:
                        st.markdown(f"""
                        <div class="ai-response">
                            <h4>🤖 AI Response:</h4>
                            <p>{result['response']}</p>
                            <small>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                        </div>
                        """, unsafe_allow_html=True)

                        # Visualization logic based on the query
                        if 'top' in user_query.lower() and 'product' in user_query.lower():
                            try:
                                limit = 5
                                if 'top 10' in user_query.lower():
                                    limit = 10
                                elif 'top 3' in user_query.lower():
                                    limit = 3
                                
                                chart_data = pd.read_sql_query(f'''
                                    SELECT msku, product_name, 
                                           SUM(total_amount) as revenue,
                                           SUM(quantity_sold) as units_sold
                                    FROM sales_records 
                                    WHERE msku IS NOT NULL
                                    GROUP BY msku 
                                    ORDER BY revenue DESC 
                                    LIMIT {limit}
                                ''', conn)
                                
                                if not chart_data.empty:
                                    fig = px.bar(
                                        chart_data, 
                                        x='product_name', 
                                        y='revenue',
                                        title=f'Top {limit} Products by Revenue',
                                        color='revenue',
                                        color_continuous_scale='viridis'
                                    )
                                    fig.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.info("Could not generate chart for this query")

                        elif 'marketplace' in user_query.lower():
                            try:
                                marketplace_data = pd.read_sql_query('''
                                    SELECT marketplace,
                                           COUNT(*) as total_orders,
                                           SUM(total_amount) as revenue
                                    FROM sales_records 
                                    GROUP BY marketplace
                                    ORDER BY revenue DESC
                                ''', conn)

                                if not marketplace_data.empty:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        fig = px.pie(
                                            marketplace_data, 
                                            values='revenue', 
                                            names='marketplace',
                                            title='Revenue by Marketplace'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col2:
                                        fig = px.bar(
                                            marketplace_data, 
                                            x='marketplace', 
                                            y='total_orders',
                                            title='Orders by Marketplace'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.info("Could not generate chart for marketplace query")
                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (30s)"):
    import time
    time.sleep(30)
    st.rerun()