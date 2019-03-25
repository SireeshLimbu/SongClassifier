5.4 Random Forest Classifier with AdaBoost
800
801 #Read Data
802 np.random.seed(1)
803 data = pd.read_csv(’Data/training_data.csv’);
804 print(data.head())
805
806 #MainData
807 X = data.drop(columns=[’label’])
808 y = data[’label’]
809
810 #Split Using Train test split
811 X_train, X_test, y_train, y_test = train_test_split(X, y,
812 test_size=0.2,random_state=0)
813
814 #Feature Scaling
815 from sklearn.preprocessing import StandardScaler
816 sc = StandardScaler()
817 X_train = sc.fit_transform(X_train)
818 X_test = sc.fit_transform(X_test)
819
820 #Fitting the model
821 rfc = RandomForestClassifier()
822 ada = AdaBoostClassifier(n_estimators=100, base_estimator = rfc,learning_rate=0.001)
823 ada.fit(X_train,y_train)
824
825 #Cross Validation and AccuracyScore
826 from sklearn.model_selection import cross_val_score
827 accuracies = cross_val_score(estimator = ada, X = X_train, y = y_train, cv = 10)
828 accuracies.mean()
829
830 #Predictions and string output for inputting in the leaderboard
831 ada_predict = ada.predict(X_test)
832 string = ’’
833 for i in ada_predict:
834 string += str(i)
835 print(string)
836
837 #Confusion Matrix and Classification Report
838 print("=== Confusion Matrix ===")
839 print(confusion_matrix(y_test, ada_predict))
840 print(’\n’)
841 print("=== Classification Report ===")
16
842 print(classification_report(y_test, ada_predict))
843 print(’\n’)
844
845 #Plotting Validation Curve
846 param_range = np.arange(1,600,50)
847 train_scores, test_scores = validation_curve(rfc,
848 X_train,
849 y_train,
850 param_name="n_estimators",
851 param_range=param_range,
852 cv=5,
853 scoring="accuracy",
854 n_jobs=-1)
855 # Calculate mean and standard deviation for training set scores
856 train_mean = np.mean(train_scores, axis=1)
857 train_std = np.std(train_scores, axis=1)
858
859 # Calculate mean and standard deviation for test set scores
860 test_mean = np.mean(test_scores, axis=1)
861 test_std = np.std(test_scores, axis=1)
862
863 # Plot mean accuracy scores for training and test sets
864 plt.plot(param_range, train_mean, label="Training score", color="black")
865 plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
866
867 # Plot accurancy bands for training and test sets
868 plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
869 color="gray")
870 plt.fill_between(param_range, test_mean - test_std, test_mean + test_std,
871 color="gainsboro")
872
873 # Create plot
874 plt.title("Validation Curve With Random Forest")
875 plt.xlabel("Number Of Trees")
876 plt.ylabel("Accuracy Score")
877 plt.tight_layout()
878 plt.legend(loc="best")
879 plt.show()
