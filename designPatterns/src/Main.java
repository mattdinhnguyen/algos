package designPatterns.src;

public class Main {
    public static void main(String[] args) {
        // User user = new User("Matthew", 30);
        // user.sayHello();
        var account = new Account();
        account.deposit(10);
        account.withdraw(5);
        System.out.println(account.getBalance());
        TaxCalculator calculator = getCalculator();
        calculator.calculateTax();
    }
    public static TaxCalculator getCalculator() {
        return new TaxCalculator2020();
    }
}
