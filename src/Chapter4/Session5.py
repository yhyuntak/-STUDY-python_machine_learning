from sklearn.ensemble import GradientBoostingClassifier
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils.util as util
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test = util.get_human_dataset()

start_time = time.time()
gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train,y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_pred=gb_pred,y_true=y_test)
print("GBM 정확도 : {0:.4f}".format(gb_accuracy))
print("수행 시간 : {0:.1f}".format(time.time()-start_time))