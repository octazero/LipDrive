import UIKit

extension UIView {
    
    // OUTPUT 1
    func dropShadow(scale: Bool = true) {
        layer.masksToBounds = false
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOpacity = 0.5
        layer.shadowOffset = CGSize(width: -1, height: 1)
        layer.shadowRadius = 2
        
        layer.shadowPath = UIBezierPath(rect: bounds).cgPath
        layer.shouldRasterize = true
        layer.rasterizationScale = scale ? UIScreen.main.scale : 1
    }
    
    // OUTPUT 2
    func dropShadow(color: UIColor, opacity: Float = 0.5, offSet: CGSize, radius: CGFloat = 1, scale: Bool = true) {
        layer.masksToBounds = false
        layer.shadowColor = color.cgColor
        layer.shadowOpacity = opacity
        layer.shadowOffset = offSet
        layer.shadowRadius = radius
        
        layer.shadowPath = UIBezierPath(rect: self.bounds).cgPath
        layer.shouldRasterize = true
        layer.rasterizationScale = scale ? UIScreen.main.scale : 1
    }
}

//Designables files are used to add extensions and properties to the parent class and add them into xcode right properties menu
@IBDesignable
class LoginCustomView: UIView {
    @IBInspectable var heightRatio: CGFloat = 1.0{
        didSet{
            draw()
        }
    }
    
    func draw() {
        let mask = CAShapeLayer()
        mask.frame = self.layer.bounds
        let width = self.layer.frame.size.width
        let height = self.layer.frame.size.height/heightRatio
        
        let path = CGMutablePath()
        path.move(to: CGPoint(x: 0, y: 0))
        path.addLine(to: CGPoint(x: width, y: 0))
        path.addLine(to: CGPoint(x: width, y: height))
        path.addLine(to: CGPoint(x: 0, y: height+50))
        
        mask.path = path
        
        let shape = CAShapeLayer()
        shape.frame = self.bounds
        shape.path = path
        shape.lineWidth = 0
        shape.strokeColor = UIColor.white.cgColor
        shape.fillColor = UIColor(red: 96/255, green: 195/255,blue: 157/255, alpha: 1.0).cgColor
        
        shape.shadowColor = UIColor(red: (96/255)*0.8, green: (195/255)*0.8,blue: (157/255)*0.8, alpha: 1.0).cgColor
        shape.shadowOffset = CGSize(width: 1, height: 1.5)
        shape.shadowOpacity = 1
        shape.shadowRadius = 3
        shape.shouldRasterize = true
        self.layer.insertSublayer(shape, at: 0)
    }
    
    
    
}
