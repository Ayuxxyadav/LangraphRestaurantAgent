i ma building a restaruant  order management   ai agent syystem using langgraph .
i will you explain very clearly what i want and what is the architecture . i will also explain you the nodes the states and the edges .
In the end i will give you the test case to simulate and tell me whether they pass or not 

The rough plan as follows :

There will be a user , when rthe cdoe runs the user will give order . this will be taken as input which i will type . This order should go the llm . The order should contain dish and required quantity .

for simplicity purpose for now we will limit the order to one particular dish and whatever quantity the user wants. 
The llm will have extract the order name and qunatity from user input .

if the user input is un related to food ordering . the llm should not give the response and tell that this is reastauarnt ordering agent not general purpose agent and will not be able to help with that .

Once the llm has order list and quantity It will send to a node called order_confirm.
The task. of order_confirm is to look at the menu ( i will tell you the menu later in this  prompt ).
Then this node will decide 3 case :
Th eorder is available 
The order is partially available ( orderr name is presnt but quantity is not matching as requirement )
The order is not available ( order name is not in menu or 0 quantiuty or negative quantity )

It will put this status of the state( I will tell you the exact content of langraph state as well  ) 

Once the llm received this 
if tghe status is confirmed ( full available with req quantity ) it should called another node called cook .
if the status is partially or not vailable it should again prompt the user to decide .
the user can either place anew order or can confirm if he wants to go ahead with the partial order .

Now when the cook node is called 

there can be 2 cases 
either cook is done then the status will bbe ready .
if the cook fails ( we can use a prob function ). give 40% of chances of failing and 60% chances of success

if the cook fails tehre should be 1 more attempt allow for cook to succed. if the cook fails evevn after these
then  the llm should issue a aplogy to the user and come to END state . 

if the cook succed , the status will be READY
and the next node will be called which is serve 

similar to cook this also has 2 cases 
serve  pass and serve fail 
thsi also has 2 retry attempt

if the serve failed 2 times then  the llm should issue a aplogy to the user and come to END state .

if the serve suuceeds the status should become complete teh llm should issue a message to the user saying your order is complete 
if servee fails then cook shoukld be called one omore time to retry.
Note that if cook has ben exhausted its retry attempt then llm hsould isuuue a apology nad come to an END state .


Now the sttae of langraph 

there should be a annoted message between llm and user 
there sahould be order deatils 
disgh name as string 
req quantityi as int 
available quantity is string  
order_confirm will write the  available quantity by reading the menu 
the llm should get to know the order confirm status by reading the state 

if dish is not availbale in menu the order_confirm  should write 0 as avavilable quantity .

 then here should be status 
 each node will update the status as specified in the above rules,
 then order retry attempt which is 3 
 cook retry attempt which is 2 
 serve retry attempt which is 2 
 
each timke failurew happened and a node is retrying . decrease the counter .
if any retry country becomes 0  it means it is over . llm should understand whetehr it has to give a retry or isuue apology by reading the counter .
in the end there should be final  result whether the order was completed or not . 

write the code and ask if there are any open questions from your side then ask i will give you some test scenarios to test the code later on .



This is Menu :
dish name: paneer butter masala, price per plate: 350 , quantity in stock: 5
dish name: dal makhani, price per plate: 250 , quantity in stock: 10
dish name: vegetable biryani, price per plate: 300 , quantity in stock: 8
dish name: chicken tikka masala, price per plate: 400 , quantity in stock: 7
dish name: garlic naan, price per plate: 50 , quantity in stock: 50
dish name: raita, price per plate: 80 , quantity in stock: 20
dish name: green salad, price per plate: 100 , quantity in stock: 15



testcase1
a user asked a unrealted question then he order a pratiall order 
and then he reject the partiall order and wants to orderr again and again he order a dish wjich is not available Expected  End due to order retry attempt 

testcase2 
user place an order which can availbale 
cook make it fails once 
cook again success
serve fails once 
serve again success
Expected End due to  order  complete 

testcase3 
user order partiall 
does not partially 
order again fully available 
cooks makes it fails 
makes again success

serve fails 
cook retry success
serve fail again

cook does not retry due to cook retry exhaust 
Overall FAIL 