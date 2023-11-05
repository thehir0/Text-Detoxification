baseline = {'ACC': 0.5797994442430833,
 'SIM': 0.530569537372181,
 'FL': 0.678929563851637,
 'J': 0.2074788008575805,
 'BLUE': 0.3868392437524953}

hypothesis1 = {'ACC': 0.7568907156673114,
 'SIM': 0.6511195927311573,
 'FL': 0.9333203696437847,
 'J': 0.4639579122112243,
 'BLUE': 0.4248203542470307}

hypothesis2 = {'ACC': 0.7727437477346865,
 'SIM': 0.7600203422316195,
 'FL': 0.913135388664151,
 'J': 0.5420948252591833,
 'BLUE': 0.5710259358110317}

hypothesis3 = {'ACC': 0.6887157182554066,
 'SIM': 0.723305512939735,
 'FL': 0.8324346852929216,
 'J': 0.4203377930078891,
 'BLUE': 0.519224402891123}

import matplotlib.pyplot as plt

# Define the metrics and hypotheses
metrics = ['ACC', 'SIM', 'FL', 'J', 'BLUE']
ACC = [baseline['ACC'], hypothesis1['ACC'], hypothesis2['ACC'], hypothesis3['ACC']]
SIM = [baseline['SIM'], hypothesis1['SIM'], hypothesis2['SIM'], hypothesis3['SIM']]
FL = [baseline['FL'], hypothesis1['FL'], hypothesis2['FL'], hypothesis3['FL']]
J = [baseline['J'], hypothesis1['J'], hypothesis2['J'], hypothesis3['J']]
BLUE = [baseline['BLUE'], hypothesis1['BLUE'], hypothesis2['BLUE'], hypothesis3['BLUE']]

index = range(len(ACC))

# Set the width of the bars

# Create a bar chart for the 'ACC' metric
plt.plot(index, ACC, label='ACC')
plt.plot(index, SIM, label='SIM')
plt.plot(index, FL, label='FL')
plt.plot(index, J, label='J')

# Set the labels for the x-axis and y-axis
plt.xlabel('Hypotheses')
plt.ylabel('Values')

# Set the title of the chart
plt.title('Models metric comparison')

# Set the x-axis labels
plt.xticks(index, ['Baseline', 'Classificator+Paraphraser', 'Pegasus', 't5-small'])

# Add a legend
plt.legend()

# Display the chart
plt.show()