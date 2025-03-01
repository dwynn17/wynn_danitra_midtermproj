import itertools
import csv
import time
import warnings
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


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


def apriori_algorithm(transactions, items, min_support, min_confidence):
    unique_items = sorted(items.keys())
    transaction_matrix = []
    
    for transaction in transactions:
        transaction_matrix.append([item in transaction for item in unique_items])

    df = pd.DataFrame(transaction_matrix, columns=unique_items, dtype=bool)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frequent_itemsets = apriori(df, min_support=min_support / 100, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence / 100)
    
    return frequent_itemsets, rules


def main():
    print("User, please select your store:")
    print("1. Amazon Store")
    print("2. Bestbuy Store")
    print("3. Kmart Store")
    print("4. Nike Store")
    print("5. General Store")
    
    store_choice = int(input("Enter your choice (1-5): "))
    
    store_files = {
        1: ("amazon_items.csv", "amazon_transactions.csv"),
        2: ("bestbuy_items.csv", "bestbuy_transactions.csv"),
        3: ("k_mart_items.csv", "k_mart_transactions.csv"),
        4: ("nike_items.csv", "nike_transactions.csv"),
        5: ("general_items.csv", "general_items_transactions.csv"),
    }

    if store_choice not in store_files:
        print("Invalid choice. Exiting.")
        return
    
    item_file, transaction_file = store_files[store_choice]

    print(f"\nYou have selected dataset located in {item_file}")
    
    min_support = int(input("Please enter the minimum support (value from 1 to 100): "))
    min_confidence = int(input("Please enter minimum confidence (value from 1 to 100): "))
    
    items = load_items(item_file)
    transactions = load_transactions(transaction_file)

    start_time = time.time()
    frequent_itemsets_brute = generate_frequent_itemsets_brute_force(items, transactions, min_support)
    brute_force_time = time.time() - start_time

    print("\nBrute Force Frequent Itemsets:")
    for k, itemsets in frequent_itemsets_brute.items():
        print(f"\n{k}-itemsets:")
        for itemset, support in itemsets:
            item_names = [items[item] for item in itemset]
            print(f"  Itemset: {item_names} - Support: {support:.2f}%")

    rules_brute = generate_association_rules(frequent_itemsets_brute, min_confidence, transactions, items)
    if not rules_brute:
        print("\nNo association rules found with the given minimum support and confidence.")
    else:
        print("\nBrute Force Association Rules:")
        for i, rule in enumerate(rules_brute, start=1):
            antecedent = [items[item] for item in rule[0]]
            consequent = [items[item] for item in rule[1]]
            print(f"\n  Rule {i}: {antecedent} => {consequent} - Confidence: {rule[2]:.2f}% - Support: {rule[3]:.2f}%")

    start_time = time.time()
    frequent_itemsets_apriori, rules_apriori = apriori_algorithm(transactions, items, min_support, min_confidence)
    apriori_time = time.time() - start_time

    print("\nApriori Frequent Itemsets:")
    for _, row in frequent_itemsets_apriori.iterrows():
        item_names = [items[item] for item in row['itemsets']]
        print(f"  Itemset: {item_names} - Support: {row['support'] * 100:.2f}%")

    print("\nApriori Association Rules:")
    for i, rule in rules_apriori.iterrows():
        antecedent = [items[item] for item in rule['antecedents']]
        consequent = [items[item] for item in rule['consequents']]
        print(f"\n  Rule {i+1}: {antecedent} => {consequent} - Confidence: {rule['confidence'] * 100:.2f}% - Support: {rule['support'] * 100:.2f}%")

    print("\nPerformance Comparison:")
    print(f"Brute Force Execution Time: {brute_force_time:.4f} seconds")
    print(f"Apriori Execution Time: {apriori_time:.4f} seconds")

    if brute_force_time < apriori_time:
        print("\nBrute Force was faster.")
    else:
        print("\nApriori was faster.")


if __name__ == "__main__":
    main()
