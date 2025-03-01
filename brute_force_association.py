import itertools
import csv

def load_items(file_name):
    items = {}
    with open(file_name, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            items[int(row[0])] = row[1]
    return items

def load_transactions(file_name):
    transactions = []
    with open(file_name, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            transactions.append([int(item) for item in row[1:]])
    return transactions

def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
    return count / len(transactions) * 100

def generate_frequent_itemsets_brute_force(items, transactions, min_support):
    frequent_itemsets = {}
    k = 1
    itemsets = [[item] for item in items.keys()]

    while itemsets:
        frequent_itemsets_k = []
        for itemset in itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets_k.append((itemset, support))

        if frequent_itemsets_k:
            frequent_itemsets[k] = frequent_itemsets_k
            itemsets = [list(combo) for combo in itertools.combinations(set(itertools.chain(*[i[0] for i in frequent_itemsets_k])), k + 1)]
            k += 1
        else:
            break

    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence, transactions, items):
    rules = []
    for k, itemsets in frequent_itemsets.items():
        for itemset, support in itemsets:
            if k > 1:
                subsets = list(itertools.chain(*[itertools.combinations(itemset, i) for i in range(1, len(itemset))]))
                for subset in subsets:
                    remainder = set(itemset) - set(subset)
                    subset_support = calculate_support(subset, transactions)
                    if subset_support == 0:
                        continue
                    confidence = (support / subset_support) * 100
                    if confidence >= min_confidence:
                        rule = (subset, remainder, confidence, support)
                        rules.append(rule)

    return rules

def main():
    print("User, please select your store:")
    print("1. Amazon Store")
    print("2. Bestbuy Store")
    print("3. Kmart Store")
    print("4. Nike Store")
    print("5. General Store")
    
    store_choice = int(input("Enter your choice (1-5): "))
    
    if store_choice == 1:
        item_file = "amazon_items.csv"
        transaction_file = "amazon_transactions.csv"
    elif store_choice == 2:
        item_file = "bestbuy_items.csv"
        transaction_file = "bestbuy_transactions.csv"
    elif store_choice == 3:
        item_file = "k_mart_items.csv"
        transaction_file = "k_mart_transactions.csv"
    elif store_choice == 4:
        item_file = "nike_items.csv"
        transaction_file = "nike_transactions.csv"
    elif store_choice == 5:
        item_file = "general_items.csv"
        transaction_file = "general_items_transactions.csv"
    else:
        print("Invalid choice. Exiting.")
        return
    
    print(f"\nYou have selected dataset located in {item_file}")
    
    min_support = int(input("Please enter the minimum support (value from 1 to 100): "))
    min_confidence = int(input("Please enter minimum confidence (value from 1 to 100): "))
    
    items = load_items(item_file)
    transactions = load_transactions(transaction_file)
    
    frequent_itemsets = generate_frequent_itemsets_brute_force(items, transactions, min_support)
    
    print("\nFrequent Itemsets:")
    for k, itemsets in frequent_itemsets.items():
        print(f"\n{k}-itemsets:")
        for itemset, support in itemsets:
            item_names = [items[item] for item in itemset]
            print(f"  Itemset: {item_names} - Support: {support:.2f}%")
    
    rules = generate_association_rules(frequent_itemsets, min_confidence, transactions, items)
    
    if not rules:
        print("\nNo association rules found with the given minimum support and confidence.")
    else:
        print("\nAssociation Rules:")
        for i, rule in enumerate(rules, start=1):
            antecedent = [items[item] for item in rule[0]]
            consequent = [items[item] for item in rule[1]]
            print(f"\n  Rule {i}: {antecedent} => {consequent} - Confidence: {rule[2]:.2f}% - Support: {rule[3]:.2f}%")

if __name__ == "__main__":
    main()