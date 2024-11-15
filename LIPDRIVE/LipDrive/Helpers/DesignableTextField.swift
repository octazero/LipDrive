import UIKit


//Designables files are used to add extensions and properties to the parent class and add them into xcode right properties menu
@IBDesignable
class DesignableTextField: UITextField {
    @IBInspectable var leftImage: UIImage?{
        didSet{
            updateView()
        }
    }
    
    @IBInspectable var leftPadding: CGFloat = 0{
        didSet{
            updateView()
        }
    }
    
    @IBInspectable var setBorder: Bool = false{
        didSet{
            if setBorder == true{
                self.setBottomBorder()
            }
        }
    }
    
    func updateView() {
        if let image = leftImage {
            leftViewMode = .always
            let imageView = UIImageView(frame: CGRect(x: leftPadding, y: 0, width: 20, height: 20))
            imageView.image = image
            
            let Width = leftPadding + 20
            
            let View = UIView(frame: CGRect(x: 0, y: 0, width: Width, height: 25))
            View.addSubview(imageView)
            
            leftView = View
        }else{
            leftViewMode = .never
        }
    }
    
    func setBottomBorder(){
        let border = CALayer()
        let width = CGFloat(1.0)
        border.borderColor = UIColor(red: 113/255, green: 167/255, blue: 146/255, alpha: 1.0).cgColor
        border.frame = CGRect(x: 10, y: self.frame.size.height - width, width:  (self.frame.size.width - 25), height: self.frame.size.height)
        
        border.borderWidth = width
        self.layer.addSublayer(border)
        self.layer.masksToBounds = true
    }
}
