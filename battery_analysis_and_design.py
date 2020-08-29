# -*- coding: utf-8 -*-
"""
Quick analysis of PV load/generation measurements with the aim of ilustrating
the trade-offs involved in the selection of a suitable battery storage system.

This script has been hacked during the Energy Data Hackdays 2020 in Brugg, 
Switzerland, see https://hack.opendata.ch/project/599.

It's in an exploratory state, there's no guarantee that anything is correct. 
Use with caution.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

# Choose one of the three provided datasets (A or B; C has different columns and
# would require some more preprocessing)
df = pd.read_csv('A.csv')

df.index = pd.to_datetime(df.Timestamp)
df.drop("Timestamp", axis=1, inplace=True)
df.rename(columns = {'Generation_kW': 'fromPV', 'Grid_Feed-In_kW': 'toGrid',
                     'Grid_Supply_kW': 'fromGrid', 
                     'Overall_Consumption_Calc_kW': 'Consumption'}, inplace=True)

power = df           # Power in kW
energy = power / 4   # Energy in kWh
energy.plot()
plt.ylabel("Energy Exchange per 15min (kWh)")


#%% Functions

def greedyBatteryDispatch(energy, batCapacity, batPower):
    """
    Naive greed battery dispatch algorithm (inefficient).
    
    Parameters
    ----------
    energy : DataFrame
        Standard dataframe for this project (enegy exchange per 15min in kWh)
        
    batCapacity : float
        Battery capacity in kWh
    
    batPower : floag
        Battery power capability in kW
    
    Returns
    -------
    DataFrame
        Standard dataframe with additional columns (SoC in kWh, toGridNew, 
                                                    fromGridNew, ...)
    """
    energy["SoC"] = 0.0 # Battery state of charge (kWh)
    energy["Charging"] = 0.0 # Effective charging (kWh)
    energy["Discharging"] = 0.0 # Effective discharging (kWh)
    
    energy["PossibleCharging"] = energy.toGrid  
    energy["PossibleCharging"].clip(lower=0.0, upper=batPower/4.0, inplace=True)
    energy["PossibleDischarging"] = -energy.fromGrid
    energy["PossibleDischarging"].clip(lower=-batPower/4.0, upper=0.0, inplace=True)
    
    for i in range(1, len(energy)):
        deltaE = 0
        if energy.SoC.values[i-1] < (batCapacity - energy.PossibleCharging.values[i]):
            deltaE =  energy.PossibleCharging.values[i]
            energy.Charging.values[i] = energy.PossibleCharging.values[i]
        
        if energy.SoC.values[i-1] > -energy.PossibleDischarging.values[i]:
            deltaE = deltaE + energy.PossibleDischarging.values[i]
            energy.Discharging.values[i] = energy.PossibleDischarging.values[i]
        
        energy.SoC.values[i] = energy.SoC.values[i-1] + deltaE 
        
    # Postprocess
    # Correct grid feed-in for energy going to or coming from the battery
    energy["toGridNew"] = energy["toGrid"] - energy["Charging"]
    energy["fromGridNew"] = energy["fromGrid"] + energy['Discharging']
    
    return energy


# Tariffs and Energy Costs

class Tariffs:
    """ Base class (interface) """
    
    supplyTariffs = 0
    feedinTariffs = 0
    
class SimpleTariffs(Tariffs):
    """ Very simply, constant tariffs. """
    
    supplyTariffs = np.ones(24)*0.23
    feedinTariffs = np.ones(24)*0.08
    
class ExampleTariffs(Tariffs):
    """ Exemplary AEW tariffs (high/low, for residential customers) """
    
    supplyTariffs = np.array([17.35, 17.35, 17.35, 17.35, 17.35, 17.35, 17.35, \
                 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36, 22.36,\
                 17.35, 17.35, 17.35, 17.35] ) * 0.01
    feedinTariffs = np.array([5.40, 5.40, 5.40, 5.40, 5.40, 5.40, 5.40, \
                 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75, 6.75,\
                 5.40, 5.40, 5.40, 5.40]) * 0.01


def calcNetEnergyCostPerYear(df, tarif, old=False):
    """
    Calculates net energy costs over one year.
    
    Paramters
    ---------
    df : DataFrame
        Standard project dataframe that already contains fromGridNew etc.
        
    tarif : Tariffs
        Tariffs class or one of it child classes
        
    old : Boolean
        Set to true if the original exchange with the grid (w/o battery) should
        be considered.
    """
    # Group by hours
    hourly = df.groupby(df.index.hour).sum()
    
    hourly["supplyTariffs"] = tarif.supplyTariffs
    hourly["feedinTariffs"] = tarif.feedinTariffs
    
    if old:
        energyCost = sum(hourly["supplyTariffs"] *    hourly["fromGrid"])
        energyRevenue = sum(hourly["feedinTariffs"] *    hourly["toGrid"])
    else:
        energyCost = sum(hourly["supplyTariffs"] *    hourly["fromGridNew"])
        energyRevenue = sum(hourly["feedinTariffs"] *    hourly["toGridNew"])
    
    netEnergyCost = energyCost - energyRevenue
    
    return netEnergyCost


# Battery Costs
class BatteryCost:
    """ Very simply battery cost model (exemplary) """
    
    def costFromCapacity(self, capacity):
        """ Returns esimtated battery costs based on the capacity"""
        return 2500.0 + 500.0 * capacity 

    def costFromCapacityOver25Years(self, capacity):
        """ Assumption: one replacement during 25 years, 40% cheaper"""
        return 1.6 * self.costFromCapacity(capacity)
        


#%% Exemplary case

testBatteryCapacity = 30.0
testBatteryPower = 10000.0

test = greedyBatteryDispatch(energy, testBatteryCapacity, testBatteryPower)

fig = plt.figure(40)
fig.clear()
ax = fig.gca()
test[["SoC", "Charging", "Discharging"]].plot(ax = ax)

# Aggregation per hour of day
hourly = test.groupby(test.index.hour).sum()

fig = plt.figure(41)
fig.clear()
ax = fig.gca()
hourly[["toGrid", "toGridNew", "fromGrid", "fromGridNew"]].plot(ax = ax)
plt.xlabel('Hour of day')
plt.ylabel("Cumulated Energy over 1 Year (kWh)")

# Energy costs before and after
netEnergyCost = calcNetEnergyCostPerYear(test, ExampleTariffs)
oldNetEnergyCost = calcNetEnergyCostPerYear(test, ExampleTariffs, old=True)

# Battery costs
bc = BatteryCost()
batteryCosts = bc.costFromCapacity(testBatteryCapacity)

# P/L 25 years
pl = bc.costFromCapacityOver25Years(testBatteryCapacity) + 25.0*(netEnergyCost-oldNetEnergyCost)
print("Example battery 25y P/L:", pl, "CHF (negative = gain)")

#%% Sweep battery ratings (grid) to generate data for contour plots

batCapRange = np.arange(0, 100, 10)
batPowerRange = np.arange(0, 21, 2.5)

selfConsumptionRatios = np.empty((len(batCapRange), len(batPowerRange)))
selfRelianceRatios = np.empty((len(batCapRange), len(batPowerRange)))
netEnergyCosts = np.empty((len(batCapRange), len(batPowerRange)))
referenceEnergyCosts = np.empty((len(batCapRange), len(batPowerRange)))

# Note: you can change the tariffs used in the last line of the for loop...

for i, batCap in enumerate(batCapRange):
    for j, batPower in enumerate(batPowerRange):
        test = greedyBatteryDispatch(energy, batCap, batPower)
        
        selfConsumptionRatios[i,j] = sum(test.Consumption - test.fromGridNew)/test.fromPV.sum()
        selfRelianceRatios[i,j] = sum(test.Consumption - test.fromGridNew)/test.Consumption.sum()
        netEnergyCosts[i,j] = calcNetEnergyCostPerYear(test, ExampleTariffs)
        referenceEnergyCosts[i,j] = calcNetEnergyCostPerYear(test, ExampleTariffs, old=True)

        
#%% Contour plots
fig = plt.figure(30)
fig.clear()
ax = fig.gca()
cs = ax.contour(batCapRange, batPowerRange, selfConsumptionRatios.T)
ax.clabel(cs, inline=1)
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Battery Power (kW)')
ax.set_title('Self-Consumption Share')

fig = plt.figure(31)
fig.clear()
ax = fig.gca()
cs = ax.contour(batCapRange, batPowerRange, selfRelianceRatios.T)
ax.clabel(cs, inline=1)
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Battery Power (kW)')
ax.set_title('Self-Reliance Share')

fig = plt.figure(32)
fig.clear()
ax = fig.gca()
cs = ax.contour(batCapRange, batPowerRange, netEnergyCosts.T)
ax.clabel(cs, inline=1)
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Battery Power (kW)')
ax.set_title('Net Energy Cost per Year')

bc = BatteryCost()
batteryCosts = np.tile(bc.costFromCapacityOver25Years(batCapRange), (len(batPowerRange),1))

# This plot shows the overall cost with the battery over the next 25 years. This 
# is different than comparing the cost of the battery vs. the reduction of 
# energy cost
fig = plt.figure(33)
fig.clear()
ax = fig.gca()
cs = ax.contour(batCapRange, batPowerRange, netEnergyCosts.T * 25.0 + batteryCosts)
ax.clabel(cs, inline=1)
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Battery Power (kW)')
ax.set_title('Overall Cost over 25 Years')

# This plot compares the P/L over 25 years (note: inverted to make profit positive)
fig = plt.figure(34)
fig.clear()
ax = fig.gca()
cs = ax.contour(batCapRange, batPowerRange, (-1.0) * ((netEnergyCosts.T-referenceEnergyCosts.T) * 25.0 \
                + batteryCosts), np.array([-30000, -20000, -10000, -2000, 0, 1000, 2000, 3000, 4000]), \
                cmap=plt.cm.coolwarm_r)
ax.clabel(cs, inline=1)
plt.xlabel('Battery Capacity (kWh)')
plt.ylabel('Battery Power (kW)')
ax.set_title('P/L over 25 years')