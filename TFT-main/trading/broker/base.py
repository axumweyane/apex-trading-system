"""
Abstract broker interface and shared data types for the trading system.
All broker implementations must conform to BaseBroker.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    STOPPED = "stopped"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


@dataclass
class AccountInfo:
    account_id: str
    status: str
    currency: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    last_equity: float
    long_market_value: float
    short_market_value: float
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionInfo:
    ticker: str
    quantity: float
    side: str
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    current_price: float
    avg_entry_price: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderInfo:
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float
    status: OrderStatus
    time_in_force: TimeInForce
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_avg_price: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRequest:
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    status: Optional[OrderStatus] = None
    message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseBroker(ABC):
    """Abstract base class for all broker implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the broker connection / session."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the broker connection / session."""

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Return current account information."""

    @abstractmethod
    async def get_positions(self) -> List[PositionInfo]:
        """Return all open positions."""

    @abstractmethod
    async def get_position(self, ticker: str) -> Optional[PositionInfo]:
        """Return position for a single ticker, or None."""

    @abstractmethod
    async def submit_order(self, request: OrderRequest) -> OrderResult:
        """Submit an order and return the result."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order."""

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Return order details by ID."""

    @abstractmethod
    async def get_open_orders(self) -> List[OrderInfo]:
        """Return all open orders."""

    @abstractmethod
    async def close_position(self, ticker: str) -> OrderResult:
        """Close a single position by ticker."""

    @abstractmethod
    async def close_all_positions(self) -> List[OrderResult]:
        """Cancel all open orders, then close all positions.
        Returns one OrderResult per closed position."""
