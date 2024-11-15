import UIKit

//Designables files are used to add extensions and properties to the parent class and add them into xcode right properties menu
@IBDesignable
class NavigationItemDesignable: UINavigationItem {
    @IBInspectable var titleLabelText: String = "Home"{
        didSet{
            ChangeLabelText()
        }
    }
    
    @IBInspectable var titleLabelColor: UIColor = UIColor.white{
        didSet{
            ChangeLabelText()
        }
    }
    
    @IBInspectable var titleLabelFont: UIFont = UIFont(name: "Comfortaa",size: 20)!{
        didSet{
            ChangeLabelText()
        }
    }
    
    @IBInspectable var titleLabelView: UIView = HomeController().getUIView(){
        didSet{
            ChangeLabelText()
        }
    }
    
    func ChangeLabelText(){
        //Change Navigation Bar to UILabel and change its attributes
        let titleLabel = UILabel(frame: CGRect(x: 0, y: 0, width: self.titleLabelView.frame.width - 32, height: 32))
        titleLabel.text = titleLabelText
        titleLabel.textColor = titleLabelColor
        titleLabel.font = titleLabelFont
        self.titleView = titleLabel
    }
}
