import numpy as np
import pandas as pd
import time
import json
import catboost
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score

class KGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, budget=1200, scorer="balanced_accuracy"):
        self.budget = budget
        self.scorer = scorer
        super().__init__()

    def fit(self, X, y):
        ### INICJALIZACJA FITA
        ###### WCIAGAMY ZBIORSONY, ŁADUJEMY MODELE, BIERZEMY ZMIENNE JAKOŚCIOWE DO CATBOOSTA
        total_start = time.time() # Na przyszłość, by sprawdzić, ile mamy czasu
        with open("models.json", "r") as configs: # Ładujemy wszystko
            models = json.load(configs)
        #print(type(X))
        categories = X.select_dtypes(include=["object", "category"]).columns
        cat_indices = X.columns.get_indexer(categories)
        crosvals = [] # Tu wrzucimy wszystkie wyniki z kroswalidacji w celu wyboru najlepszego modelu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        ### POJEDYŃCZE WYKONANIE WSZYSTKIEGO POZA PĘTLĄ - W CELU USTALENIA ILE MOŻEMY ZAPUŚCIĆ MODELI

        #start = time.time() 
        #cls = eval(models[0]["class"]) # robimy zaciąg z napisu
        #obj = Pipeline([
        #    ("impute", SimpleImputer(strategy="most_frequent")),
        #    ("catpred", cls(cat_features = cat_indices, **models[0]["params"]))]) # używamy zaciągu jako modelu (catboost)
        #obj.fit(X_train, y_train)
        #sc = balanced_accuracy_score(y_test, obj.predict(X_test)) # wynik na kroswalidacji
        #print(f"model 0: {sc}")
        #crosvals.append(sc) # dodawańsko
        #end = time.time()
        ### UŻYWAMY METODY ROZSĄDNEJ, ABY WYBRAĆ, NA ILE MODELI STARCZY NAM CZASU

        #benchmark = end-start
        #print(f"time elapsed: {benchmark}")
        #approx_list = 15*[benchmark] + 15*[2*benchmark] + 15*[20*benchmark] + [100*benchmark] # więcej estymatorów -- więcej czasu
        #upper_bound = sum(np.cumsum(approx_list)<(self.budget*3/4))-1 # działa. NIE TYKAJCIE TEGO, pozostałe modele będą się liczyły w tzw marginesie
        # margines - 1/4 budżetu (ustalone metodą bo tak)
        #print(f"Setting upper bound for {upper_bound} with 1/4 time in reserve")

        ### ROBIMY WSZYSTKO JEDNYM WHILEM

        i = 0 
        while (self.budget - time.time() + total_start)>(self.budget/4) and (i<45):
            start = time.time() 
            cls = eval(models[i]["class"])
            obj = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("catpred", cls(cat_features = cat_indices, **models[i]["params"]))])
            obj.fit(X_train, y_train)
            sc = balanced_accuracy_score(y_test, obj.predict(X_test))
            crosvals.append(sc)
            end = time.time() 
            print(f"model {i}: {sc:.4f}, Elapsed time: {(end - start):.4f} seconds")
            print(f"Time left: {(self.budget - time.time() + total_start):.4f} seconds")

            i+=1


        ### pętla for na benchmarku, spróbujemy whilem to zrobić
        #for i in range(1, upper_bound): 
        #    cls = eval(models[i]["class"])
        #    obj = Pipeline([
        #    ("impute", SimpleImputer(strategy="most_frequent")),
        #    ("catpred", cls(cat_features = cat_indices, **models[i]["params"]))])
        #    obj.fit(X_train, y_train)
        #    sc = balanced_accuracy_score(y_test, obj.predict(X_test))
        #    crosvals.append(sc)
        #    print(f"model {i}: {sc}")


        best_model_index = crosvals.index(max(crosvals)) # wybieramy naszego szefa
        best_estimator = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("catpred", eval(models[best_model_index]["class"])(cat_features = cat_indices, **models[best_model_index]["params"]))]) # budujemy szefa
        ### TUTAJ ZAJMIEMY SIĘ ENSEMBLAMI -- STACKING, VOTING, CO KTO CHCE, NAJWYŻEJ ZMIENIMY MARGINES
        # Murzynie jak dodasz swoje modele to wjeb tu jakiś stacking, co wybiera z modeli nie catboostowych jako pomoce
        # w stackingu finalny model będzie najlepszym catboostem
        estimators_list_catboost = [(models[i]["name"]+"_"+str(i),eval(models[i]["class"])(cat_features = cat_indices, **models[i]["params"])) for i in random.sample(range(1, i-1), 5)]
        # możemy przypadkiem wybrać najlepszy model jako jeden do stackowania, jest mi strasznie wszystko jedno, mamy 4 inne, także jebać

        start = time.time()
        stack_boost = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("catpred", StackingClassifier(estimators = estimators_list_catboost, 
                                           final_estimator=eval(models[best_model_index]["class"])(cat_features = None, **models[best_model_index]["params"])))]) #stacking jeden jest zrobiony
        stack_boost.fit(X_train, y_train)
        sc_stack_boost = balanced_accuracy_score(y_test, stack_boost.predict(X_test)) #testujemy na kroswalce
        print(f"catboost_stack: {sc_stack_boost}")
        end = time.time() # badamy jak szybko się zensembli to całe dziadowstwo

        rough_benchmark = end - start
        rough_remainder = self.budget - (end - total_start)
        ### NA PRZYSZŁOŚĆ - JEŚLi zostaje nam więcej budżetu niż rough_benchmark, to robimy kolejny ensembel. Jeśli nie, to ucinamy i tyle

        # Wybieramy i zwracamy do selfa szefa totalnego
        # Trzeba go ofc zafitować, bo nigdy tego nie robiliśmy lol
        if sc_stack_boost > max(crosvals):
            stack_boost.fit(X,y)
            self.best_model = stack_boost
        else:
            best_estimator.fit(X, y)
            self.best_model = best_estimator

        self.is_fitted_ = True

        return self
    


    ### Dwie proste funkcje na predict i predict_proba
    def predict(self, X):
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        return self.best_model.predict_proba(X)


        



        
        
