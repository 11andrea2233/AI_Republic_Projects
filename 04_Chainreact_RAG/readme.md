ChainReact README
Introduction

Welcome to ChainReact, an interactive supply chain management tool designed to help you optimize and streamline your logistics and operations. Whether you're managing production, overseeing transportation, or ensuring timely order fulfillment, ChainReact offers data-driven insights and practical solutions to improve the efficiency of your supply chain.
How It Works

ChainReact enables users to input and analyze data across five core components of the supply chain:

    Sourcing and Procurement: Manage supplier relationships, procurement strategies, and cost analysis.
    Production and Manufacturing: Streamline production processes, optimize batch sizes, and monitor quality control.
    Inventory Management: Track stock levels, set reorder points, and optimize inventory to prevent stockouts or overstocking.
    Transportation and Distribution: Optimize transportation costs, plan efficient routes, and reduce delivery times.
    Order Fulfillment: Enhance order processing, improve shipping timelines, and track fulfillment status.

The program analyzes real-time data, identifies inefficiencies, and provides actionable insights to help you make smarter decisions.
Ideal Users

    Supply Chain Managers: Looking to optimize logistics and production processes.
    Logistics Coordinators: Seeking solutions for transportation and distribution optimization.
    Data Analysts: Interested in using data to improve inventory management, production planning, and order fulfillment.
    Small to Medium Enterprises (SMEs): Looking to scale their supply chain operations with efficiency and ease.

Whether you're managing a small local supply chain or a large global network, ChainReact provides you with tools to enhance your operations and make informed decisions faster.
Dataset Overview

The ChainReact program utilizes datasets for each core component of the supply chain. These datasets are designed to help users simulate real-world scenarios, identify key metrics, and generate insights for optimization.
1. Sourcing and Procurement Dataset

This dataset helps manage supplier relationships and procurement strategies. It includes details such as supplier IDs, ingredient types, costs, and lead times.
Supplier_ID	Supplier_Name	Ingredient	Unit_Cost	Lead_Time (days)
S001	Fresh Farms	Potatoes	$0.10/lb	7
S002	Spice Masters	Salt	$0.02/lb	5
S003	Pure Oil Co.	Oil	$0.50/lb	10
S004	Snack Packagers	Packaging	$0.05/unit	3
S005	Green Farms	Spices	$0.20/lb	14
2. Production and Manufacturing Dataset

Track production processes, including batch sizes, production times, and quality checks. This dataset helps identify efficiency bottlenecks and optimize manufacturing workflows.
Batch_ID	Product_ID	Batch_Size (lbs)	Production_Time (hrs)	Quality_Check	Cost
B001	P001	500	5	Pass	$200
B002	P002	600	6	Pass	$240
B003	P003	450	4	Fail	$180
B004	P004	550	5	Pass	$220
B005	P005	500	5	Pass	$210
3. Inventory Management Dataset

This dataset helps track stock levels across multiple warehouse locations, set reorder points, and monitor days to reorder to avoid stockouts.
Item_ID	Warehouse_Location	Stock_Level (lbs)	Reorder_Point (lbs)	Days_to_Reorder
I001	WH-Chicago	1,200	300	4
I002	WH-LA	950	250	5
I003	WH-Dallas	1,500	500	3
I004	WH-New York	800	200	6
I005	WH-Seattle	600	150	4
4. Transportation and Distribution Dataset

This dataset provides transportation details such as shipment modes, transit times, origin and destination locations, and shipping costs, enabling users to optimize their distribution strategies.
Shipment_ID	Mode_of_Transport	Origin	Destination	Transit_Time (days)	Cost
T001	Truck	Chicago, IL	Milwaukee, WI	1	$150
T002	Rail	LA, CA	Houston, TX	3	$500
T003	Air	Miami, FL	NYC, NY	2	$700
T004	Truck	Seattle, WA	San Francisco, CA	1	$200
T005	Sea	Houston, TX	Vancouver, CA	5	$800
5. Order Fulfillment Dataset

This dataset tracks order statuses, customer IDs, shipping dates, and product quantities, helping users monitor and optimize their fulfillment processes.
Order_ID	Customer_ID	Product_ID	Quantity (lbs)	Order_Status	Shipping_Date
O001	C001	P001	150	Shipped	2023-10-01
O002	C002	P002	200	Pending	2023-10-02
O003	C003	P003	100	Delivered	2023-10-01
O004	C004	P004	120	Shipped	2023-10-03
O005	C005	P005	180	In Progress	2023-10-04
Getting Started

    Install Dependencies: Before running ChainReact, ensure you have the required Python libraries installed. You can install them via pip:

pip install pandas numpy matplotlib

Run the Program: Start by loading your supply chain data into the program. You can input the sample datasets provided or use your own real-world data. ChainReact will guide you through data analysis and help you generate actionable insights for optimizing your supply chain processes.