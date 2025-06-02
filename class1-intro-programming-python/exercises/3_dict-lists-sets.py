#Now we will read some data in from a file and use dictionaries, lists, and sets to store the data in a variable

#####################################################################################
#here is some basic code for reading lines from a file
#you can see the contents of the file in 'data.txt'
#the for loop allows us to work with each line in the data file, in this case fruits
#The '.strip()' removes all newline characters so we can work with just the strings

for line in open('2_data.txt'):
  print(line.strip())

#####################################################################################

#EXERCISE 1: modify the above code to save the fruits into a list and print the list

fruits_list = []

for line in open('data.txt'):
  pass #TO DO: write code so that the data is saved in a list

print("List of fruits: ", fruits_list)




#EXERCISE 2: modify the code to save the fruits into a set

fruits_set = {}

for line in open('data.txt'):
  pass #TO DO: write code so that the data is saved in a set

print("Set of fruits: ", fruits_set)




#EXERCISE 3 (CHALLENGE): modify the code to save the fruits in a dictionary where each key is a fruit and each value is the number of times that fruit appears in the data 

fruits_dict = {}

for line in open('data.txt'):
  pass #TO DO: write code so that the data is saved into a dict

print("Dict of fruits: ", fruits_dict)
