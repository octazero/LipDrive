import UIKit


class HistoryPageViewController: UIViewController, UITableViewDelegate, UITableViewDataSource{
    @IBOutlet weak var HistoryTableView: UITableView!
    
    let HistoryLabels = ["History 1","History 2","History 3","History 4","History 5","History 6","History 7","History 8","History 9"]
    let HistoryTime = ["10:01 AM","10:02 AM","10:03 AM","10:04 AM","10:05 AM","10:06 AM","10:07 AM","10:08 AM","10:09 AM"]
    let HistoryDate = ["1/1/2018","1/1/2018","1/1/2018","1/1/2018","1/1/2018","1/1/2018","1/1/2018","1/1/2018","1/1/2018"]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //change the background color of the table
        self.HistoryTableView.backgroundColor = UIColor(red: 96/255, green: 195/255, blue: 157/255, alpha: 1.0)
        self.HistoryTableView.separatorStyle = UITableViewCellSeparatorStyle.none
        self.HistoryTableView.separatorColor = UIColor.white
    }
    
    //This predefined function is used to initialize the number of rows in the table
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.HistoryLabels.count
    }
    
    //this function initialize the first cell with the history cells
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let Cell = Bundle.main.loadNibNamed("HistoryTableViewCell", owner: self, options: nil)?.first as! HistoryTableViewCell
        
        Cell.HistoryLabel.text = HistoryLabels[indexPath.row]
        Cell.TimeLabel.text = HistoryTime[indexPath.row]
        Cell.DateLabel.text = HistoryDate[indexPath.row]
        return Cell
    }

    //onPressing the table cell
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        //self.SelectedRow = indexPath.row
        //self.HistoryTableView.reloadData()
    }
}
