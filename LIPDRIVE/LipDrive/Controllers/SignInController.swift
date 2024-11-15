import UIKit


class SignInController: UIViewController, UITextFieldDelegate{
    @IBOutlet weak var UsernameAndPasswordView: UIView!
    @IBOutlet weak var usernameTextField: UITextField!
    @IBOutlet weak var passwordTextField: UITextField!
    @IBOutlet weak var SignUpButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //initialize corner radius for the view that contains the textfields
        self.UsernameAndPasswordView.layer.cornerRadius = 15
        self.UsernameAndPasswordView.clipsToBounds = true
        
        //adding shade to the signUp button
        let Multiplier = 0.5
        self.SignUpButton.layer.shadowOffset = CGSize(width: 2, height: 2)
        self.SignUpButton.layer.shadowRadius = 5
        self.SignUpButton.layer.shadowColor = UIColor(red: CGFloat((96/255)*Multiplier), green: CGFloat((195/255)*Multiplier),blue: CGFloat((157/255)*Multiplier), alpha: 0.6).cgColor
        self.SignUpButton.layer.shadowOpacity = 1
        
        //Rounding the sign up button
        self.SignUpButton.layer.cornerRadius = 20
        
        //initialize tags for textfields to be able to switch between them by pressing next in keyboard
        self.usernameTextField.tag = 0
        self.passwordTextField.tag = 1
    }
    
    //This function closes the keyboard by pressing anywhere on the screen
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.view.endEditing(true);
    }
    
    //This function is called by pressing on the return value of any keyboard
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        if let nextField = textField.superview?.viewWithTag(textField.tag + 1) as? UITextField {
            nextField.becomeFirstResponder()
        } else {
            textField.resignFirstResponder()
        }
        return false
    }

}
