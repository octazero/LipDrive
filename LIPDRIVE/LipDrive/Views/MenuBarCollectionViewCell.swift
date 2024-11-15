import UIKit

class MenuBarCollectionViewCell: UICollectionViewCell {
    @IBOutlet weak var myImage: UIImageView!
    
    //this function is used to add white line under the selected menu label
    func setupHorizontalBar(){
        let border = CAShapeLayer()
        border.backgroundColor = UIColor.white.cgColor
        border.frame = CGRect(x: 0, y: self.frame.size.height-5, width: self.frame.size.width, height: 5)
        UIView.animate(withDuration: 1) {
            border.setAffineTransform(CGAffineTransform(translationX: 0, y: 0))
        }
        border.name = "Border"
        self.layer.addSublayer(border)
    }
    
    //this function is used to remove white line under the selected menu label
    func removeHorizontalBar(){
        for layer in self.layer.sublayers! {
            if layer.name == "Border" {
                layer.removeFromSuperlayer()
            }
        }
    }
}
