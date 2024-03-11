### SA-1 ASSIGNMENT
### Developed By : Tirupathi Jayadeep
### Register Number : 212223240169
### Dept : AIML


### Objective 1 :
### To Create a scatter plot between cylinder vs Co2Emission (green color).
### Code:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()
```
### Output:
![image](https://github.com/23004426/ML-WORKSHOP/assets/144979327/9b32cf6c-ec0b-4ca4-961d-4ee459eb2ae1)


### Objective 2 :
### Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors.
### Code:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='red', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='yellow', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()
```
### Output:
![image](https://github.com/23004426/ML-WORKSHOP/assets/144979327/213b2d1d-f11e-4cd4-b434-bda4d84d547c)


### Objective 3 :
### Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors.
### Code:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='brown', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='green', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()
```
### Output:
![image](https://github.com/23004426/ML-WORKSHOP/assets/144979327/9ea2f30f-3f5c-41dc-aeca-7a7bdc4543a6)

### Objective 4 :
### Train your model with independent variable as cylinder and dependent variable as Co2Emission.
### Code:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FuelConsumption.csv')

X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']

X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)

model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)

```
### Output:
![image](https://github.com/23004426/ML-WORKSHOP/assets/144979327/e417c31b-3543-4a48-8b8d-1a39504a79e0)


### Objective 5 :
### Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission.
### Code:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FuelConsumption.csv')

X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']

X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)
```
### Output:
![image](https://github.com/23004426/ML-WORKSHOP/assets/144979327/67689119-674e-4930-9927-916cd610f859)


### Objective 6 :
### Train your model on different train test ratio and train the models and note down their accuracies.
### Code:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
ratios = [0.2, 0.4, 0.6, 0.8]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')
```
### Output:
![image](https://github.com/23004426/ML-WORKSHOP/assets/144979327/c846ef5e-fd53-4cc5-8df7-b129bcfd43b4)

### Result : 
All the programs executed successfully.

