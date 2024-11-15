import UIKit

class SettingsPageViewController: UIViewController,UITableViewDataSource,UITableViewDelegate {
    @IBOutlet weak var SettingsTableView: UITableView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //change the background color of the table
        self.SettingsTableView.backgroundColor = UIColor(red: 96/255, green: 195/255, blue: 157/255, alpha: 1.0)
    }
    
    //This predefined function is used to initialize the number of rows in the table
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return 2
    }
    
    //this function initialize the first cell with the clear history and profile details cells
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        if indexPath.row == 1{
            let Cell = Bundle.main.loadNibNamed("ClearHistoriesTableViewCell", owner: self, options: nil)?.first as! ClearHistoriesTableViewCell
            return Cell
        }else{
            let Cell = Bundle.main.loadNibNamed("ProfileTableViewCell", owner: self, options: nil)?.first as! ProfileTableViewCell
            Cell.ProfilePictureImageView.image = UIImage(named: "1")
            Cell.NameLabel.text = "Karim Azmi"
            Cell.EmailLabel.text = "Kariimazmi@gmail.com"
            return Cell
        }
        
        let Cell = Bundle.main.loadNibNamed("ProfileTableViewCell", owner: self, options: nil)?.first as! ProfileTableViewCell
        return Cell
    }
    
    //onPressing the table cell
    func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
        if indexPath.row == 1{
            return 56
        }else{
            return 117
        }
    }
}
