import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


ds = pd.read_csv('../../data/student_sleep_patterns.csv')
print('Кількість колонок - ', len(ds.columns), '\n')
print('Колонки: ', ds.columns, '\n')

#Перевірка на пропущені значення в датасеті
print('Пропущені значення: ', '\n', pd.isnull(ds).any())

#Перевірка на типи даних в датасеті
ds.info()


# Середнє значення для конкретного стовпця (наприклад, "Sleep_Quality")
mean_value = ds['Sleep_Quality'].mean()
print("Середнє значення для 'Sleep_Quality':", mean_value)

# Середнє значення для всіх числових стовпців
mean_values = ds.mean()
print("Середнє значення для всіх числових стовпців:\n", mean_values)


plt.rcParams['figure.figsize'] = [10, 5]
sns.countplot(x='University_Year', hue='Sleep_Quality', data=ds, palette='dark')
plt.xlabel('University Year')
plt.ylabel('Sleep Quality')
plt.title('Sleep Quality & University Year')
plt.legend(title='Sleep Quality')
plt.show()

plt.rcParams['figure.figsize'] = [10, 5]
sns.countplot(x='Caffeine_Intake', hue='Sleep_Quality', data=ds, palette='dark')
plt.xlabel('Caffeine_Intake')
plt.ylabel('Sleep Quality')
plt.title('Sleep Quality Distribution')
plt.legend(title='Sleep quality', loc='upper right')
plt.show()

plt.rcParams['figure.figsize'] = [10, 5]
sns.countplot(x='Gender', hue='Sleep_Quality', data=ds, palette='dark')
plt.xlabel('Gender')
plt.ylabel('Sleep Quality')
plt.title('Sleep Quality Distribution')
plt.legend(title='Sleep quality', loc='upper right')
plt.show()

plt.rcParams["figure.figsize"] = (10,5)
plt.title("Age & Sleep Quality", fontsize=20)
graph = ds[["Age", "Sleep_Quality"]]
ax = sns.barplot(x="Age", y="Sleep_Quality", data=graph)
plt.show()

def diagnostic_plots(df, variable):

        # define figure size
        plt.figure(figsize=(16, 4))

        # histogram
        plt.subplot(1, 3, 1)
        sns.histplot(df[variable], bins=30)
        plt.title('Histogram')

        # Q-Q plot
        plt.subplot(1, 3, 2)
        stats.probplot(df[variable], dist="norm", plot=plt)
        plt.ylabel('Variable quantiles')

        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=df[variable])
        plt.title('Boxplot')

        plt.show()

diagnostic_plots(ds, 'Age')
diagnostic_plots(ds, 'Screen_Time')
diagnostic_plots(ds, 'Sleep_Duration')
diagnostic_plots(ds, 'Study_Hours')
diagnostic_plots(ds, 'Physical_Activity')
diagnostic_plots(ds, 'Caffeine_Intake')

df = pd.read_csv('../../data/encoded_data.csv')

correlation_matrix = df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix')
plt.show()






