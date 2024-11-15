import UIKit

class ForgotPasswordController: UIViewController, UITextFieldDelegate {
    @IBOutlet weak var ChangePasswordFields: UIView!
    @IBOutlet weak var EmailField: DesignableTextField!
    @IBOutlet weak var OldPasswordField: DesignableTextField!
    @IBOutlet weak var NewPasswordField: DesignableTextField!
    @IBOutlet weak var ConfirmNewPasswordField: DesignableTextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //initialize corner radius for the view that contains the textfields
        self.ChangePasswordFields.layer.cornerRadius = 15
        self.ChangePasswordFields.clipsToBounds = true
        
        //initialize tags for textfields to be able to switch between them by pressing next in keyboard
        self.EmailField.tag = 0
        self.OldPasswordField.tag = 1
        self.NewPasswordField.tag = 2
        self.ConfirmNewPasswordField.tag = 3
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
    
    @IBAction func backButton(_ sender: UIButton) {
        self.dismiss(animated: true, completion: nil)
    }
}
