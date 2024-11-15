import UIKit

//This is related to the XIB file
class HomePageViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    //this function is used to open the camera viewcontroller by pressing the lip logo
    @IBAction func LipButtonIsPressed(_ sender: UIButton) {
        let storyBoard = UIStoryboard(name: "Main",bundle: nil)
        let CameraFrameViewController = storyBoard.instantiateViewController(withIdentifier: "CameraViewController") as! CameraViewController
        self.present(CameraFrameViewController, animated: true, completion: nil)
    }
}
