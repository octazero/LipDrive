import UIKit

class SignUpController: UIViewController, UITextFieldDelegate{
    @IBOutlet weak var SignUpView: UIView!
    @IBOutlet weak var FirstNameField: DesignableTextField!
    @IBOutlet weak var LastNameField: DesignableTextField!
    @IBOutlet weak var EmailField: DesignableTextField!
    @IBOutlet weak var PasswordField: DesignableTextField!
    @IBOutlet weak var ConfirmPasswordField: DesignableTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //initialize corner radius for the view that contains the textfields
        self.SignUpView.layer.cornerRadius = 15
        self.SignUpView.clipsToBounds = true
        
        //initialize tags for textfields to be able to switch between them by pressing next in keyboard
        self.FirstNameField.tag = 0
        self.LastNameField.tag = 1
        self.EmailField.tag = 2
        self.PasswordField.tag = 3
        self.ConfirmPasswordField.tag = 4
    }
    
    @IBAction func BackButton(_ sender: UIButton) {
        self.dismiss(animated: true, completion: nil)
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
